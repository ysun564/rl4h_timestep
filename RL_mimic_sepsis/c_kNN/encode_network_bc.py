"""
Create and apply a behavior-cloning (BC) network to saved episodic datasets.

The script loads a BC model from the calibration folder, loads episodic
datasets that contain encoded states and KNN probabilities, computes BC
probabilities for each state, assigns those probabilities to both the
`estm_pibs` and `pibs` entries in the dataset, and saves the updated
datasets to a new folder named `episodes+encoded_state+bc_pibs`.

The script processes timesteps 1, 2, 4, and 8 by default.
"""
from pathlib import Path
import json
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_mlp(state_dimension, hidden_dimension, action_dimension):
    """Return a simple two-layer MLP mapping states to action logits.

    Inputs are raw state vectors. The final layer returns logits and does
    not include a softmax so that caller can convert to probabilities.
    """
    return nn.Sequential(
        nn.Linear(state_dimension, hidden_dimension),
        nn.ReLU(),
        nn.Linear(hidden_dimension, hidden_dimension),
        nn.ReLU(),
        nn.Linear(hidden_dimension, action_dimension),
    )

def load_metadata(metadata_path):
    """Load JSON metadata from the given path and return it as a dict."""
    with metadata_path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def load_bc_model(model_directory):
    """Load a BC model from model_directory and return an evaluative model.

    The function attempts to load a scripted model first. If that fails
    it reads metadata.json, constructs a compatible MLP, and loads a
    state dictionary from model_best.pt. The returned model is on CPU
    and set to evaluation mode.
    """
    model_directory = Path(model_directory)
    model_file = model_directory / 'model_best.pt'
    metadata_file = model_directory / 'metadata.json'

    # Try to load a scripted model first.
    try:
        model = torch.jit.load(str(model_file), map_location='cpu')
        model.eval()
        return model
    except Exception:
        pass

    # Load checkpoint and construct model from metadata if needed.
    if not metadata_file.exists():
        raise FileNotFoundError('Missing metadata.json in %s.' % str(model_directory))

    metadata = load_metadata(metadata_file)
    state_dim = metadata.get('state_dim')
    action_dim = metadata.get('action_dim')
    hidden_dim = metadata.get('hidden_dim', 256)

    checkpoint = torch.load(str(model_file), map_location='cpu')

    # Determine whether checkpoint is a full model or state dict.
    if isinstance(checkpoint, torch.nn.Module):
        checkpoint.eval()
        return checkpoint

    # If checkpoint is a dict, try to find a state dict inside it.
    state_dict = None
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            # Heuristic: if values are tensors then treat object as state dict.
            if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                state_dict = checkpoint

    if state_dict is None:
        raise RuntimeError('Could not interpret model file at %s.' % str(model_file))

    model = build_mlp(state_dim, hidden_dim, action_dim)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def compute_probabilities_for_episode_states(states_tensor, model):
    """Compute action probabilities for a batch of state vectors.

    The input is a torch tensor shaped (T, state_dim). The function returns
    a torch tensor shaped (T, action_dim) containing probabilities.
    """
    with torch.no_grad():
        logits = model(states_tensor.float())
        probabilities = F.softmax(logits, dim=-1)
    return probabilities


def assign_bc_probs_to_dataset(dataset, model):
    """Assign BC probabilities into dataset['pibs'] and ['estm_pibs'].

    The function follows the same indexing convention used in the
    notebook: for each episode i, it computes probabilities for
    states up to length-1 and assigns them into the time-aligned slots.
    """
    # Create containers if they do not exist.
    if 'actionvecs' not in dataset:
        raise KeyError('Dataset missing actionvecs key.')

    dataset['pibs'] = torch.zeros_like(dataset['actionvecs'])
    dataset['estm_pibs'] = torch.zeros_like(dataset['actionvecs'])

    # Iterate episodes and fill in probabilities up to the episode length.
    num_episodes = len(dataset['icustayids'])
    for i in range(num_episodes):
        length = int(dataset['lengths'][i])
        if length <= 1:
            # No time steps to predict for very short episodes.
            continue

        # Extract states for which to compute BC probabilities.
        states = dataset['statevecs'][i][: length - 1].cpu()
        probabilities = compute_probabilities_for_episode_states(states, model)

        # Convert to same dtype and assign into both fields.
        dataset['pibs'][i, : length - 1, :] = probabilities
        dataset['estm_pibs'][i, : length - 1, :] = probabilities


def process_timestep(data_root, model_root, timestep):
    """Process a single timestep setting: load model, assign BC probs, save data.

    The function expects bc models in `model_root/bc_network_dt{timestep}h` and
    input datasets inside `data_root/data_asNormThreshold_dt{timestep}h`.
    The processed datasets are written to a new directory named
    `episodes+encoded_state+bc_pibs` inside the same data folder.
    """
    model_directory = Path(model_root) / f'bc_network_dt{timestep}h'
    if not model_directory.exists():
        raise FileNotFoundError('Model directory not found: %s' % str(model_directory))

    model = load_bc_model(model_directory)

    data_directory = Path(data_root) / f'data_asNormThreshold_dt{timestep}h' / 'episodes+encoded_state+knn_pibs_k5sqrtn_uniform'
    if not data_directory.exists():
        raise FileNotFoundError('Data directory not found: %s' % str(data_directory))

    save_directory = Path(data_root) / f'data_asNormThreshold_dt{timestep}h' / 'episodes+encoded_state+bc_pibs'
    save_directory.mkdir(parents=True, exist_ok=True)

    # Load each dataset split if present, process and save.
    for split in ('train_data.pt', 'val_data.pt', 'test_data.pt'):
        input_file = data_directory / split

        dataset = torch.load(str(input_file), map_location='cpu')
        assign_bc_probs_to_dataset(dataset, model)

        output_file = save_directory / split
        torch.save(dataset, str(output_file))


def main():
    """Main entry point to process all default timesteps.

    The script assumes a repository layout where the bc networks live in
    RL_mimic_sepsis/c_kNN/calibration/bc_network and the datasets live in
    RL_mimic_sepsis/data. The base paths are resolved relative to this
    script file location for convenience.
    """
    base_path = Path(__file__).resolve().parents[1]
    data_root = base_path / 'data'
    model_root = base_path / 'c_kNN' / 'calibration' / 'bc_network'

    # Process these timesteps by default.
    timesteps = (1, 2, 4, 8)
    for t in timesteps:
        try:
            process_timestep(data_root, model_root, t)
        except Exception as exc:  # pragma: no cover
            # Report errors but continue with remaining timesteps.
            print('Failed to process dt%dh: %s' % (t, str(exc)))


if __name__ == '__main__':
    main()
