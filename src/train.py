import pytorch_lightning as pl
import datasets
import models
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)

if __name__ == "__main__":
    main_dm = datasets.MainDataModule(
        datasets.TrainDataSet("data/ru_small.bin", "data/sample_locs.npy", 16),
        train_bsz=64,
    )

    encoder_model = models.EncoderModel(vocab_size=64000, learning_rate=1)  # 4e-4

    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_callback = ModelCheckpoint(
        dirpath="data/checkpoints/", save_top_k=4, monitor="val_loss", save_last=True
    )

    trainer = pl.Trainer(
        # fast_dev_run=1,
        # limit_train_batches=100,
        max_epochs=500,
        accelerator="gpu",
        devices=[4, 5, 6, 7],
        strategy=DDPStrategy(static_graph=True, find_unused_parameters=False),
        callbacks=[lr_monitor, checkpoint_callback],
        log_every_n_steps=200,
        precision=16,
        limit_val_batches=1500,
        val_check_interval=0.5,
        max_time="00:09:00:00",
        # accumulate_grad_batches=16,
        gradient_clip_val=1,
        # check_val_every_n_epoch=1,
        enable_checkpointing=True,
    )

    trainer.fit(
        model=encoder_model,
        datamodule=main_dm,
        # ckpt_path="/home/is/armin-sa/Projects/lm/data/checkpoints/epoch=0-step=100.ckpt",
    )
