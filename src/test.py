import torch
import pytorch_lightning as pl
import models as models
import datasets as datasets

checkpoint_path = "./e9.ckpt"
test_data_path = "test_id.txt"

decoder_model = models.DecoderModel(vocab_size=64000).load_from_checkpoint(
    checkpoint_path, strict=False
)


decoder_trainer = pl.Trainer(
    # limit_test_batches=0.1,
    accelerator="gpu",
    devices=1,
)

decoder_trainer.test(
    model=decoder_model,
    dataloaders=torch.utils.data.DataLoader(
        datasets.DecoderDataSet(test_data_path, is_npy=False, all=True)
    ),
)
