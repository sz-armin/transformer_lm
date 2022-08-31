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
        datasets.TrainDataSet("data/ru_small.bin", "data/sample_locs.npy", 8), train_bsz=8
    )
    test_ds = datasets.TestDataSet()
    test_dl = DataLoader(test_ds, batch_size=8)

    # baseline_model = models.BaselineModel()
    encoder_model = models.EncoderModel(vocab_size=8000, learning_rate=1e-2)

    trainer = pl.Trainer(
        # fast_dev_run=1,
        limit_train_batches=10000,
        max_epochs=2,
        accelerator="gpu",
        auto_lr_find=True,
        devices=1,
        # strategy=DDPStrategy(find_unused_parameters=True),
    )

    # lr_finder = trainer.tuner.lr_find(encoder_model, datamodule=main_dm, num_training=1000)
    # print(lr_finder.suggestion())
    # fig = lr_finder.plot(suggest=True)
    # fig.savefig("1.png")

    # trainer.fit(model=baseline_model, datamodule=main_dm)
    trainer.fit(model=encoder_model, datamodule=main_dm)
    # trainer.fit(model=encoder_model, train_dataloaders=test_dl)

    with torch.no_grad():
        tmp = encoder_model(torch.tensor([[1]]))
        # tmp = encoder_model(torch.tensor([[5,4,4,2]]))
    print(tmp.argmax())
    # print(tmp[108])
    print(tmp)

    with torch.no_grad():
        tmp = encoder_model(torch.tensor([[1, 479, 1301, 70, 99]]))
        # tmp = encoder_model(torch.tensor([[5,4,4,2]]))
    print(tmp.argmax())
    # print(tmp[835])
    print(tmp)

    with torch.no_grad():
        tmp = encoder_model(torch.tensor([[1, 479, 1301, 70, 99, 835, 7]]))
        # tmp = encoder_model(torch.tensor([[5,4,4,2]]))
    print(tmp.argmax())
    # print(tmp[49])
    print(tmp)