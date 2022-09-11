import pytorch_lightning as pl
import datasets
import models, random
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)

class Randomizer(pl.Callback):
    def on_train_epoch_end(self, trainer, model):
        start_seed = random.randint(0, trainer.datamodule.train_dataset.context + 1)
        trainer.datamodule.train_dataset.start_seed=start_seed

if __name__ == "__main__":
    main_dm = datasets.DecoderDataModule(
        datasets.DecoderDataSet("data/ru_small_id.txt"),
        datasets.DecoderDataSet("data/dev_id.txt"),
        train_bsz=16, num_workers=8
    )

    decoder_model = models.DecoderModel(vocab_size=64000, learning_rate=1)

    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_callback = ModelCheckpoint(
        dirpath="data/checkpoints_dec/", save_top_k=4, monitor="val_loss", save_last=True
    )

    decoder_trainer = pl.Trainer(
        # fast_dev_run=1,
        max_epochs=10,
        accelerator="gpu",
        devices=[0,1,2,3],
        strategy=DDPStrategy(static_graph=True, find_unused_parameters=False),
        # callbacks=[lr_monitor, checkpoint_callback, Randomizer()],
        callbacks=[lr_monitor, Randomizer()],
        log_every_n_steps=50,
        precision=16,
        # limit_val_batches=1500,
        val_check_interval=0.5,
        # max_time="00:38:30:00",
        gradient_clip_val=1,
        enable_checkpointing=False,
        # limit_train_batches=10,
        # check_val_every_n_epoch=50
    )

    decoder_trainer.fit(
        model=decoder_model,
        datamodule=main_dm,
        # ckpt_path="/home/is/armin-sa/Projects/lm/data/checkpoints/epoch=4-step=162534.ckpt",
    )
