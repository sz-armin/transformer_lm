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
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    StochasticWeightAveraging,
    ModelCheckpoint,
)

if __name__ == "__main__":
    main_dm = datasets.MainDataModule(
        datasets.TrainDataSet("data/ru_small.bin", "data/sample_locs.npy", 16),
        train_bsz=64,
    )
    # test_ds = datasets.TestDataSet()
    # test_dl = DataLoader(test_ds, batch_size=256, num_workers=8)

    encoder_model = models.EncoderModel(vocab_size=64000, learning_rate=1) #4e-4

    lr_monitor = LearningRateMonitor(logging_interval="step")
    # checkpoint_callback = ModelCheckpoint(
    #     dirpath="data/checkpoints/", save_top_k=4, monitor="val_loss"
    # )

    trainer = pl.Trainer(
        # fast_dev_run=1,
        limit_train_batches=9999999,
        max_epochs=20,
        accelerator="gpu",
        devices=[4, 5, 6, 7],
        strategy=DDPStrategy(find_unused_parameters=True),
        # callbacks=[lr_monitor, checkpoint_callback],
        callbacks=[lr_monitor],
        log_every_n_steps=100,
        precision=16,
        limit_val_batches=1000,
        val_check_interval=0.5,
        max_time="00:8:00:00",
        # accumulate_grad_batches=16,
        gradient_clip_val=0.5,
        check_val_every_n_epoch=10,
        enable_checkpointing=False,
    )

#     lr_finder = trainer.tuner.lr_find(encoder_model, datamodule=main_dm)

#     fig = lr_finder.plot(suggest=True)
#     print(lr_finder.suggestion()
# )
#     fig.savefig("lr_find_plot.png")

    trainer.fit(
        model=encoder_model,
        datamodule=main_dm,
        # ckpt_path="/home/is/armin-sa/Projects/lm/data/checkpoints/epoch=1-step=4507.ckpt",
    )
    # trainer.fit(model=encoder_model, train_dataloaders=test_dl)
