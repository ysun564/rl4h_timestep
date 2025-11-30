import time

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from test_tube import Experiment, HyperOptArgumentParser, SlurmCluster

from model import AIS_GRU
from data import MIMIC3SepsisDataModule, timestep


def main(hparams, cluster):
    # Wait for Slurm file writes to propagate.
    time.sleep(15)
    pl.seed_everything(42)
    
    dm = MIMIC3SepsisDataModule()
    model = AIS_GRU(
        dm.observation_dim,
        dm.context_dim,
        dm.num_actions, 
        latent_dim=hparams.latent_dim,
        lr=hparams.lr,
    )
    logger = CSVLogger(save_dir=('/local/scratch/ysun564/project/OfflineRL_TimeStep'
                                 '/RL_mimic_sepsis/b_train_encoder/logs'), 
                       name=f'AIS_GRU_model_asNormThreshold_dt{timestep}h_grid',
                       version = f'dt{timestep}h_lr{hparams.lr}_latent{hparams.latent_dim}')

    trainer = pl.Trainer(
        logger=logger,
        accelerator='auto', 
        devices='auto',
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=50, verbose=False),
            ModelCheckpoint(monitor='train_loss'),
            ModelCheckpoint(monitor='val_loss'),
            ModelCheckpoint(monitor='val_mse'),
        ],
    )
    trainer.fit(model, dm)

def optimize_on_cluster(hyperparams):
    # enable cluster training
    cluster = SlurmCluster(
        hyperparam_optimizer=hyperparams,
        log_path='/local/scratch/ysun564/project/OfflineRL_TimeStep/RL_mimic_sepsis/slurm',
    )

    # Email for cluster coms.
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
    

    # Creates and submits jobs to slurm.
    cluster.optimize_parallel_cluster_gpu(
        main,
        # Creates 30 trials.
        nb_trials=30,
        job_name=f'mimic_sepsis_AIS_GRU_job_asNormThreshold_dt{timestep}h',
    )   

if __name__ ==  '__main__':
    # Hyperparameter search grid
    parser = HyperOptArgumentParser(strategy='grid_search')
    parser.opt_list('--latent_dim', default=32, type=int, tunable=True, options=[8, 16, 32, 64, 128])
    parser.opt_list('--lr', default=1e-4, type=float, tunable=True, options=[1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 5e-4])
    
    hparams = parser.parse_args()
    optimize_on_cluster(hparams) 
