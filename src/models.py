from turtle import pos
import torch
from torch import nn
import pytorch_lightning as pl
import math

class LearnedPosEmbedding(nn.Module):
    def __init__(self, num_embeddings=1026, embedding_dim=512):
        self.num_embeddings = num_embeddings + 1
        self.new_pad_idx = self.num_embeddings - 1
        super().__init__()
        self.embedding = nn.Embedding(
            self.num_embeddings, embedding_dim, padding_idx=self.new_pad_idx
        )

    def forward(self, x, padding_idx=3):
        pos = [
            [self.new_pad_idx for _ in range(len(row[row == 3]))]
            + list(range(len(row[row != 3] + 1)))
            for row in x
        ]
        pos = torch.tensor(pos, device=x.device)

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
    def __init__(self, learning_rate=1e-3, d_model=1024, vocab_size=8000):
        super().__init__()
        self.learning_rate = learning_rate
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, self.d_model, padding_idx=3)
        self.pos_emb = LearnedPosEmbedding(self.vocab_size, self.d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            dim_feedforward=2048,
            nhead=8,
            norm_first=False,
            batch_first=True,
        )
        self.layernorm=nn.LayerNorm(self.d_model)
        self.encoder = nn.TransformerEncoder(
            self.encoder_layer, norm=self.layernorm, num_layers=6
        )
        self.linear = nn.Sequential(nn.Linear(self.d_model, self.vocab_size))
        # self.w = nn.Parameter(
        #     torch.rand(self.embedding.weight.shape), requires_grad=True
        # )  # why can't we reuse the weights?

    def forward(self, x):
        mask = (x == 3)
        # print(x)
        # mask = None
        emb= self.embedding(x) + self.pos_emb(x)
        # print(x[0], emb[0])
        res = self.encoder(
                emb,
                src_key_padding_mask=mask
            )
        
        res[mask]=0
        res = res.sum(-2)
        # res = res[..., -1, :]
        # print(res)

        res = self.linear(res)
        return res

    def training_step(self, batch, batch_idx):
        x, y = batch
        # y = y.unsqueeze(-1).float()
        # x = batch[:, :-1]
        # y = batch[:, -1]
        # print(x, y)

        # y = nn.functional.one_hot(
        #     y.clone().detach().long(), num_classes=self.vocab_size
        # ).float()
        y=y.long()
        # print(y)
        y_hat = self(x)
        # print(y_hat[0][y[0].argmax()], y[0].argmax())

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

    # def validation_step(self, batch, batch_idx):
    #     x, y = batch
    #     y = nn.functional.one_hot(
    #         y.clone().detach().long(), num_classes=self.vocab_size
    #     ).float()
    #     y_hat = self(x)
    #     loss = nn.functional.cross_entropy(y_hat, y, reduction="mean")

    #     self.log("val_loss", loss, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     optimizer, T_0=75, T_mult=2
        # )
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]
