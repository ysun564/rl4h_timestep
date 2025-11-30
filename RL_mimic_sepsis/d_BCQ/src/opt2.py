"""Conduct BCQ grid search. 

NOTE:
1. Change timestep in model.py
2. /local/scratch/ysun564/anaconda3/envs/rl4h_rep_new/bin/python /local/scratch/ysun564/project/OfflineRL_TimeStep/RL_mimic_sepsis/d_BCQ/src/opt2.py
"""

knn_pib = False

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import argparse

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from test_tube import HyperOptArgumentParser, SlurmCluster

from data import EpisodicBuffer, SASRBuffer
from data import add_data_specific_args, remap_rewards

if knn_pib:
    from model import timestep, action_space, num_actions, state_dim, horizon
    from model import BCQ
else:
    from model_old2 import timestep, action_space, num_actions, state_dim, horizon
    from model_old2 import BCQ

def main(args, cluster):
    # Wait for Slurm file writes to propagate.
    time.sleep(15)

    # Seed & Logging
    pl.seed_everything(args.seed)
    logger = CSVLogger(save_dir=('/local/scratch/ysun564/project/OfflineRL_TimeStep'
                                 '/RL_mimic_sepsis/d_BCQ/logs/BCQ_bcnet'),
                       name=f'BCQ_as{action_space}_dt{timestep}h_grid_bcnet',
                       version = f'dt{timestep}_threshold{args.threshold}seed{args.seed}')

    # Load training and validation data.
    train_buffer = SASRBuffer(state_dim, num_actions)
    train_buffer.load('/local/scratch/ysun564/project/OfflineRL_TimeStep/RL_mimic_sepsis'
                     f'/data/data_as{action_space}_dt{timestep}h'
                     f'/episodes+encoded_state+knn_pibs_k5sqrtn_uniform/train_data.pt')
    
    val_episodes = EpisodicBuffer(state_dim, num_actions, horizon)
    val_episodes.load('/local/scratch/ysun564/project/OfflineRL_TimeStep/RL_mimic_sepsis'
                     f'/data/data_as{action_space}_dt{timestep}h'
                      '/episodes+encoded_state+knn_pibs_k5sqrtn_uniform/val_data.pt')

    # Remap reward from [0,-1,1] to [R_immed, R_death, R_disch]
    train_buffer.reward = remap_rewards(train_buffer.reward, args)
    val_episodes.reward = remap_rewards(val_episodes.reward, args)
    Rmin = float(np.min(train_buffer.reward.cpu().numpy()))
    Rmax = float(np.max(train_buffer.reward.cpu().numpy()))
    
    # Data loaders for train and val
    train_buffer_loader = DataLoader(train_buffer, batch_size=100, shuffle=True)
    val_episodes_loader = DataLoader(val_episodes, batch_size=len(val_episodes), shuffle=False)

    # Create model
    policy = BCQ(
        state_dim=state_dim,
        num_actions=num_actions,
        Rmin=Rmin,
        Rmax=Rmax,
        **vars(args),
    )
    
    # Create trainer with custom validation logic
    trainer = pl.Trainer(
        accelerator="auto", 
        logger=logger,
        val_check_interval=args.eval_frequency,
        max_steps=args.max_steps,
        callbacks=[
            ModelCheckpoint(save_top_k=-1, filename='{step:04d}', every_n_train_steps=100, save_on_train_epoch_end=False),
            TQDMProgressBar(),
        ],
    )
    
    # Run trainer
    trainer.fit(policy, train_buffer_loader, val_episodes_loader)

def optimize_on_cluster(hyperparams):
    # enable cluster training
    cluster = SlurmCluster(
        hyperparam_optimizer=hyperparams,
        log_path='/local/scratch/ysun564/project/OfflineRL_TimeStep/RL_mimic_sepsis/slurm/BCQ/BCQ_bcnet_job',
    )

    # email for cluster coms
    cluster.notify_job_status(email='ysun564@emory.edu', on_done=False, on_fail=False)
    
    # Configure cluster.
    cluster.per_experiment_nb_gpus = 1
    cluster.per_experiment_nb_nodes = 1
    # Configure maximum time.
    cluster.job_time = '12:00:00'
    cluster.memory_mb_per_node = 8000
    cluster.add_command('eval "$(/local/scratch/ysun564/anaconda3/bin/conda shell.bash hook)"')
    cluster.add_command('conda activate rl4h_rep_new')
    cluster.add_slurm_cmd('partition', 'hopper', comment='')
    cluster.add_slurm_cmd('ntasks', '1', comment='')
    cluster.add_slurm_cmd('cpus-per-task', '2', comment='')
    cluster.add_slurm_cmd('gpus', '1', comment='') 

    # creates and submits jobs to slurm
    cluster.optimize_parallel_cluster_cpu(
        main,
        nb_trials=1,
        job_name=f'mimic_sepsis_dBCQ_job_as{action_space}_dt{timestep}h',
    )

if __name__ ==  '__main__':
    # Hyperparameter search grid
    parser = HyperOptArgumentParser(strategy='grid_search')
    parser = add_data_specific_args(parser)
    parser.add_argument('--max_steps', type=int, default=10_000)  # Max number of training steps
    parser.add_argument('--eval_frequency', type=int, default=100) # Check validation every N training steps
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--eval_discount", type=float, default=1.0)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument('--target_value_clipping', default=False, action=argparse.BooleanOptionalAction)
    parser.opt_list("--seed", type=int, tunable=True, options=[4])
    parser.opt_list("--threshold", type=float, tunable=True, options=[0.9999])
    
    hparams = parser.parse_args()
    optimize_on_cluster(hparams)