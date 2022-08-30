import torch
from torch import nn
import pytorch_lightning as pl

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
            sync_dist=True
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
    def __init__(self, d_emb=512, vocab_size=32000):
        super().__init__()
        self.d_emb = d_emb
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, self.d_emb)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, dim_feedforward=2048, nhead=8, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)

        self.w = nn.Parameter(
            torch.zeros(self.embedding.weight.shape), requires_grad=True
        )  # why can't we reuse the weights?

    def forward(self, x):
        res = self.encoder(self.embedding(x)).mean(-2) @ self.w.T

        return res

    def training_step(self, batch, batch_idx):
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
            sync_dist=True
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