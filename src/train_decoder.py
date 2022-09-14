import pytorch_lightning as pl
import datasets
import models, random
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    DeviceStatsMonitor,
)


class Randomizer(pl.Callback):
    def on_train_epoch_end(self, trainer, model):
        start_seed = random.randint(0, trainer.datamodule.train_dataset.context)
        trainer.datamodule.train_dataset.start_seed = start_seed


if __name__ == "__main__":
    main_dm = datasets.DecoderDataModule(
        datasets.DecoderDataSet("data/ru_small_id.npy", True),
        datasets.DecoderDataSet("data/dev_id.txt"),
        train_bsz=70,
        num_workers=16,
    )

    decoder_model = models.DecoderModel(vocab_size=64000, learning_rate=2e-4)

    lr_monitor = LearningRateMonitor(logging_interval="step")
    device_stats = DeviceStatsMonitor()
    checkpoint_callback = ModelCheckpoint(
        dirpath="data/checkpoints_dec/",
        save_top_k=2,
        monitor="val_loss",
        save_last=True,
        save_on_train_epoch_end=True,
        every_n_epochs=1,
    )

    decoder_trainer = pl.Trainer(
        # fast_dev_run=1,
        max_epochs=10,
        accelerator="gpu",
        devices=[4, 5, 6, 7],
        strategy="deepspeed_stage_2",
        callbacks=[lr_monitor, checkpoint_callback, Randomizer(), device_stats],
        # callbacks=[lr_monitor, Randomizer()],
        log_every_n_steps=100,
        precision=16,
        val_check_interval=0.05,
        # max_time="00:38:30:00",
        gradient_clip_val=1,
        enable_checkpointing=True,
        reload_dataloaders_every_n_epochs=1,
    )

    decoder_trainer.fit(
        model=decoder_model,
        datamodule=main_dm,
        # ckpt_path="/home/is/armin-sa/Projects/lm/data/checkpoints_dec/last.ckpt",
    )
