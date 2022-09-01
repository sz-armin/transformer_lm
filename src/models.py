from turtle import pos
import torch
from torch import nn
import pytorch_lightning as pl
import math

from torch.optim.lr_scheduler import _LRScheduler


class LearnedPosEmbedding(nn.Module):
    def __init__(self, num_embeddings=1026, embedding_dim=512):
        self.num_embeddings = num_embeddings + 1
        self.internal_pad_idx = 0
        super().__init__()
        self.embedding = nn.Embedding(
            self.num_embeddings,
            embedding_dim,
            padding_idx=self.internal_pad_idx,
        )

    def forward(self, x, padding_idx=3):
        mask = x.ne(padding_idx)
        pos = torch.cumsum(mask, dim=1)

        return self.embedding(pos)


class BaselineModel(pl.LightningModule):
    def __init__(self, d_emb=512, vocab_size=32000):
        super().__init__()
        self.d_emb = d_emb
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, self.d_emb)
        self.encoder = nn.Sequential(
            nn.Linear(self.d_emb, 2048), nn.Tanh(), nn.Linear(2048, self.d_emb)
        )
        self.w = nn.Parameter(
            torch.zeros(self.embedding.weight.shape), requires_grad=True
        )  # why can't we reuse the weights?

    def forward(self, x):
        res = self.encoder(self.embedding(x)).mean(-2) @ self.w.T

        return res

    def training_step(self, batch, batch_idx):
        # x, y = batch[:,:-1], batch[:,-1]
        # y = y.unsqueeze(-1).float()

        x, y = batch

        y = nn.functional.one_hot(
            y.clone().detach().long(), num_classes=self.vocab_size
        ).float()
        y_hat = self(x)

        loss = nn.functional.cross_entropy(y_hat, y, reduction="mean")

        self.log("train_loss", loss, logger=True)
        self.log(
            "train_loss_epoch",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = nn.functional.one_hot(
            y.clone().detach().long(), num_classes=self.vocab_size
        ).float()
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y, reduction="mean")

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optimizer


class EncoderModel(pl.LightningModule):
    def __init__(
        self,
        learning_rate=1e-3,
        d_model=1024,
        vocab_size=8000,
        dropout=0.1,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.warmup = 4000
        self.factor = 1
        self.dropout_rate = dropout
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.dropout = nn.Dropout(self.dropout_rate)
        self.layernorm = nn.LayerNorm(self.d_model)

        self.embedding = nn.Embedding(
            self.vocab_size, self.d_model, padding_idx=3)
        self.pos_emb = LearnedPosEmbedding(self.vocab_size, self.d_model)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            dim_feedforward=self.d_model * 4,
            nhead=8,
            dropout=self.dropout_rate,
            norm_first=True, # Disable Fast Path for AMP compatibility
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            self.encoder_layer,
            norm=self.layernorm,
            num_layers=8,
            # enable_nested_tensor=True,
        )

        self.linear = nn.Linear(self.d_model, self.vocab_size)

    def forward(self, x):
        mask = (x == 3)

        x = self.dropout(self.embedding(x)) + self.dropout(self.pos_emb(x))
        x = self.encoder(x, src_key_padding_mask=mask)
        x = self.dropout(x)

        x[mask] = 0

        x = x.sum(-2)
        x = self.linear(x)

        return x

    def training_step(self, batch, batch_idx):
        if True:
            x, y = batch
        else:
            x = batch[:, :-1]
            y = batch[:, -1]

        y_hat = self(x)

        loss = nn.functional.cross_entropy(y_hat, y, reduction="mean")

        self.log("train_loss", loss, logger=True)
        self.log("bsz", float(x.shape[0]), logger=True)
        self.log("wpb", float(x.shape[0]*x.shape[1]), logger=True)

        self.log(
            "train_loss_epoch",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.long()
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y, reduction="mean")

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=1e-4
        )
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     optimizer, T_0=2000, T_mult=2
        # )
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer, lr_lambda=self.rate
        )

        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    def rate(self, step):
        if step == 0:
            step = 1
        return self.factor * (
            self.d_model ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5))
        )
