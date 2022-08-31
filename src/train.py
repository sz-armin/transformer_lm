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
from pytorch_lightning.callbacks import LearningRateMonitor

if __name__ == "__main__":
    main_dm = datasets.MainDataModule(
        datasets.TrainDataSet("data/ru_small.bin", "data/sample_locs.npy", 16), train_bsz=8
    )
    # test_ds = datasets.TestDataSet()
    # test_dl = DataLoader(test_ds, batch_size=256, num_workers=8)

    # baseline_model = models.BaselineModel()
    encoder_model = models.EncoderModel(vocab_size=8000, learning_rate=1e-4)

    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        # fast_dev_run=1,
        limit_train_batches=100,
        max_epochs=1,
        accelerator="gpu",
        auto_lr_find=True,
        devices=2,
        strategy=DDPStrategy(find_unused_parameters=True),
        callbacks=[lr_monitor],
        log_every_n_steps=10,
        # profiler="advanced",
        # precision=16,
        # num_sanity_val_steps=0
    )

    # lr_finder = trainer.tuner.lr_find(encoder_model, datamodule=main_dm, num_training=1000)
    # print(lr_finder.suggestion())
    # fig = lr_finder.plot(suggest=True)
    # fig.savefig("1.png")

    # trainer.fit(model=baseline_model, datamodule=main_dm)
    trainer.fit(model=encoder_model, datamodule=main_dm)
    # trainer.fit(model=encoder_model, train_dataloaders=test_dl)