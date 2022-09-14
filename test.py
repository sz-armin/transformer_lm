from turtle import pos
import torch
from torch import nn
import pytorch_lightning as pl
import math
import src.models as models
import src.datasets as datasets

decoder_model = models.DecoderModel(vocab_size=64000).load_from_checkpoint("/home/is/armin-sa/Projects/lm/e5.ckpt", strict=False)


decoder_trainer = pl.Trainer(
    limit_test_batches=0.001,
    accelerator="gpu",
    devices=1,
    )

decoder_trainer.test(
        model=decoder_model,
        dataloaders=torch.utils.data.DataLoader(datasets.DecoderDataSet("data/ru_small_id-3.npy", True)),
        # ckpt_path="/home/is/armin-sa/Projects/lm/final.ckpt",
    )
