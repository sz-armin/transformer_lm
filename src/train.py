import torch, sys, random
from torch import nn
import pytorch_lightning as pl
from pathlib import Path
import numpy as np
from copy import deepcopy
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from typing import Optional
import datasets
import models
from pytorch_lightning.strategies import DDPStrategy

if __name__ == "__main__":
    main_dm = datasets.MainDataModule(
        datasets.TrainDataSet("data/ru_small.bin", "data/sample_locs.npy", 8), train_bsz=1
    )
    # test_ds = datasets.TestDataSet()
    # test_dl = DataLoader(test_ds, batch_size=8)

    # baseline_model = models.BaselineModel()
    encoder_model = models.EncoderModel()

    trainer = pl.Trainer(
        limit_train_batches=10000,
        max_epochs=2,
        accelerator="gpu",
        devices=3,
        strategy=DDPStrategy(find_unused_parameters=False),
    )
    # trainer.fit(model=baseline_model, datamodule=main_dm)
    trainer.fit(model=encoder_model, datamodule=main_dm)
