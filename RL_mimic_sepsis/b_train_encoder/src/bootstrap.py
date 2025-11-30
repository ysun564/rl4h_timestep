"""Generate the bootstrapped AIS encoder.
TODO: Adjust 'timestep' and 'local' in data.py.
"""
import os
import torch
import numpy as np
from numpy.random import default_rng

import pytorch_lightning as pl

from data import MIMIC3SepsisDataModule, timestep
from model import AIS_GRU

model_dir = (rf'F:\time_step\OfflineRL_FactoredActions\RL_mimic_sepsis\b_train_encoder'
             rf'\logs\AIS_GRU_model_asNormThreshold_dt{timestep}h_best_128')

# Find the only .ckpt file in that directory.
ckpt_files = [f for f in os.listdir(model_dir) if f.endswith(".ckpt")]
if len(ckpt_files) != 1:
    raise ValueError(f"Expected exactly one .ckpt file, found {len(ckpt_files)}: {ckpt_files}")

ckpt_path = os.path.join(model_dir, ckpt_files[0])

def main():
    dm = MIMIC3SepsisDataModule()
    dm.setup("validate")

    model = AIS_GRU.load_from_checkpoint(ckpt_path)

    print("► HParams in checkpoint:")
    for key, value in model.hparams.items():
        print(f"  {key}: {value}")
    print()

    trainer = pl.Trainer(accelerator="gpu", devices=1, logger=False)

    val_metrics = trainer.validate(
        model=model,
        dataloaders=dm.val_dataloader(),
        ckpt_path=None,  
    )
    print(val_metrics)

    errs = trainer.predict(model, dataloaders=dm.val_dataloader())
    if errs is None:
        errs = []
    flat_errs = []
    for result in errs:
        if result is None:
            continue
        if isinstance(result, list):
            flat_errs.extend(result)
        else:
            flat_errs.append(result)
    errors = torch.cat(flat_errs).cpu().numpy()  # flat vector of all valid squared errors
    print(errors.mean())  # should match val_mse now

    rng = default_rng(42)
    boot = [
        rng.choice(errors, size=len(errors), replace=True).mean()
        for _ in range(1000)
    ]
    lower, upper = np.percentile(boot, [2.5, 97.5])
    print(f"MSE = {errors.mean():.5f}  [95% CI: {lower:.5f}, {upper:.5f}]")


if __name__ == "__main__":
    main()