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
from pytorch_lightning.callbacks import LearningRateMonitor, StochasticWeightAveraging

if __name__ == "__main__":
    main_dm = datasets.MainDataModule(
        datasets.TrainDataSet("data/ru_small.bin", "data/sample_locs.npy", 4), train_bsz=1
    )
    # test_ds = datasets.TestDataSet()
    # test_dl = DataLoader(test_ds, batch_size=256, num_workers=8)

    encoder_model = models.EncoderModel(vocab_size=16000, learning_rate=1e-0)

    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        # fast_dev_run=1,
        limit_train_batches=9999999,
        max_epochs=1,
        accelerator="gpu",
        auto_lr_find=True,
        devices=4,
        strategy=DDPStrategy(find_unused_parameters=True),
        callbacks=[lr_monitor],
        # callbacks=[lr_monitor, StochasticWeightAveraging(swa_lrs=1e-5)],
        log_every_n_steps=250,
        # profiler="advanced",
        precision=16,
        # detect_anomaly=True,
        limit_val_batches=1000,
        val_check_interval=0.02
    )

    trainer.fit(model=encoder_model, datamodule=main_dm)
    # trainer.fit(model=encoder_model, train_dataloaders=test_dl)