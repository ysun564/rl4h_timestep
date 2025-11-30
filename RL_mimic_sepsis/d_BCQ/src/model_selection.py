'''This script is for rerunning the model selection.
NOTE: We should execute the following commands to run the script as a module.
r`D:\\Software\\anaconda3\\envs\\rl4h_rep_new\\python.exe -m RL_mimic_sepsis.d_BCQ.src.model_selection`
'''

import os
import sys
import pandas as pd
import numpy as np
import torch
import copy
from pathlib import Path
from types import SimpleNamespace
from torch.utils.data import DataLoader
from joblib import Parallel, delayed

sys.path.append(r'F:\time_step\OfflineRL_FactoredActions')
# sys.path.append(r'/local/scratch/ysun564/project/OfflineRL_TimeStep')
from RL_mimic_sepsis.utils.timestep_util import (get_horizon, get_state_dim,
                                                 timestep_list, action_space_list)
from model_old import BCQ
from data import remap_rewards
from evaluate import EpisodicBufferO

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def load_val_dataset(timestep, state_dim, num_actions, horizon, action_space):
    """
    """
    val_episodes_O = EpisodicBufferO(state_dim, num_actions, horizon)

    val_episodes_O.load(rf'F:/time_step/OfflineRL_FactoredActions/RL_mimic_sepsis/data/data_as{action_space}_dt{timestep}h'       
        rf'/episodes+encoded_state+bc_pibs/val_data.pt')

    
    val_episodes_O.reward = remap_rewards(
        val_episodes_O.reward, 
        SimpleNamespace(**{'R_immed': 0.0, 'R_death': 0.0, 'R_disch': 100.0}))
    
    tmp_val_episodes_loader_O = DataLoader(val_episodes_O, batch_size=len(val_episodes_O), shuffle=False)
    val_batch_O = next(iter(tmp_val_episodes_loader_O))

    return val_batch_O


def evaluate_checkpoint(ckpt_path, _shared_val_batch, clipping_value: float, eps=0.1):
    """Evaluate one checkpoint and return 5 values.
    clipping_value: For 'offline_evaluation'.
    """
    print(f'Path of loaded checkpoint: {ckpt_path}')
    val_batch = copy.deepcopy(_shared_val_batch)
    
    fname = os.path.basename(ckpt_path)    
    name, _ = os.path.splitext(fname)     
    iteration = int(name.split("=", 1)[1].split("-", 1)[0])   

    with torch.no_grad():
        # Load model on CPU.
        model_1 = BCQ.load_from_checkpoint(checkpoint_path=ckpt_path, map_location='cpu')
        model_1.eval()

        # Move model to GPU for inference.
        device = torch.device("cuda:0")
        model_1  = model_1.to(device)
    
        # Also move validation batch to GPU since it is tensor.
        val_batch_gpu = [
            t.to(device, non_blocking=True) for t in val_batch
        ]

        wis_est, wis_ess = model_1.offline_evaluation(
            val_batch_gpu, eps=eps,
            clipping=clipping_value
            )
        qvalues = model_1.offline_q_evaluation(val_batch_gpu)

    print({
    "iteration":    iteration,
    "step":         iteration - 1,
    "val_wis":      wis_est,
    "val_qvalues":  qvalues,
    "val_ess":      wis_ess,
    })

    return {
    "iteration":    iteration,
    "step":         iteration - 1,
    "val_wis":      wis_est,
    "val_qvalues":  qvalues,
    "val_ess":      wis_ess,
    }


def rebuild_metrics(val_batch, base_dir: Path, metrics_name: str, clipping_value: float, parallel: bool = True):
    """ I don't like this function because it is too long to understand.
    TODO: Move the name of new csv file to the 
    """
    base = base_dir
    use_cols = ["iteration", "step", "val_wis", "val_qvalues", "val_ess"]
    for version_dir in sorted(base.glob("dt*")):
        
        if not version_dir.is_dir():
            continue
        print(f"→ Processing {version_dir.name}")

        old_csv = version_dir / "metrics.csv"
        new_csv = version_dir / metrics_name

        # 1) write header row
        pd.DataFrame(columns=use_cols).to_csv(new_csv, index=False)

        # 2) grab all checkpoints and sort by the number after “=” and before “-”
        ckpt_dir = version_dir / "checkpoints"
        ckpt_files = sorted(
            ckpt_dir.glob("step=*-v1.ckpt"),
            key=lambda p: int(p.stem.split("=", 1)[1].split("-", 1)[0])
        )
        if parallel:
            rows = Parallel(n_jobs=8)(delayed(evaluate_checkpoint)(ckpt, val_batch, clipping_value) for ckpt in ckpt_files)
        else:
            # TODO: This line is hard to read and should be rewritten.
            rows = [evaluate_checkpoint(ckpt, val_batch, clipping_value) for ckpt in ckpt_files]

        # for ckpt in ckpt_files:
        #     metrics = evaluate_checkpoint(ckpt, val_batch)
        #     rows.append(metrics)

        df = pd.DataFrame(list(rows), columns=use_cols)
        df.to_csv(version_dir / metrics_name, index=False)
        print(f" Wrote {new_csv} ({len(ckpt_files)} entries)")

if __name__ == '__main__':

    # Each time we should adjust 'run_type'.
    run_type = 'folder'

    

    if run_type == 'grid':
        timestep_list = [1]
        action_space_list = ['NormThreshold']
        clipping_value = 1.438
        
        # For grid selection, run across time steps and action spaces.
        for timestep in timestep_list:
            for action_space in action_space_list:
                num_actions = 25
                state_dim = get_state_dim(timestep, action_space)
                horizon = get_horizon(timestep)

                base_dir = Path(rf'F:\time_step\OfflineRL_FactoredActions\RL_mimic_sepsis\d_BCQ\logs'
                                rf'\BCQ_bcnet\BCQ_as{action_space}_dt{timestep}h_grid_bcnet')
                print(base_dir)

                val_batch_O = load_val_dataset(timestep, state_dim, num_actions, horizon, action_space)
                rebuild_metrics(val_batch_O, base_dir, 'test.csv', clipping_value, parallel = False)
                
    if run_type == 'one':
        # Check one specific checkpoint file. 
        # Each time adjust 'ckpt_path', 'timestep', 'action_space' and 'step'.
        # That defines the file that will be loaded.

        ckpt_path = (rf'F:/time_step/OfflineRL_FactoredActions/RL_mimic_sepsis/d_BCQ/logs/BCQ_nips/BCQ_asNormThreshold_dt1h_grid/dt1_threshold0.0seed0/checkpoints/step=0100-v1.ckpt')
        timestep = 2
        action_space = 'NormThreshold'
        clipping_value = 1.6
        state_dim = 128
        eps = 0.1

        val_dataset = load_val_dataset(timestep, state_dim, 25, get_horizon(timestep), action_space)

        evaluate_checkpoint(ckpt_path, val_dataset, clipping_value, eps=eps)
    
    if run_type == 'folder':
        folder_path = Path(rf'F:\time_step\OfflineRL_FactoredActions\RL_mimic_sepsis\d_BCQ\logs\BCQ_bcnet\BCQ_asNormThreshold_dt4h_grid_bcnet\dt4_threshold0.5seed0\checkpoints')
        timestep = 8

        action_space = 'NormThreshold' 
        clipping_value = 1.438
        state_dim = 128
        val_dataset = load_val_dataset(timestep, state_dim, 25, get_horizon(timestep), action_space)
        # Gather all checkpoint files.
        checkpoint_paths = list(folder_path.glob("step=*.ckpt"))

        # Extract the step number from a filename like "step=0100-v1.ckpt".
        def get_step_number(path):
            stem = path.stem  # "step=0100-v1"
            after_equal = stem.split("=", 1)[1]  # "0100-v1"
            step_str = after_equal.split("-", 1)[0]  # "0100"
            return int(step_str)

        # Sort checkpoints by their step number.
        checkpoint_paths.sort(key=get_step_number)

        # Evaluate each checkpoint in order.
        for ckpt_path in checkpoint_paths:
            metrics = evaluate_checkpoint(ckpt_path, val_dataset, clipping_value)

            # Append metrics to ../metric_clip{clipping_value}.csv (one row per checkpoint)
            out_csv = (folder_path.parent / f"metric_clip{clipping_value}.csv")
            use_cols = ["iteration", "step", "val_wis", "val_qvalues", "val_ess"]
            row_df = pd.DataFrame([metrics], columns=use_cols)

            if not out_csv.exists():
                row_df.to_csv(out_csv, index=False)
            else:
                row_df.to_csv(out_csv, mode="a", header=False, index=False)





    


            
