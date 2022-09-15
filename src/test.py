import torch
import pytorch_lightning as pl
import src.models as models
import src.datasets as datasets

checkpoint_path = "/home/is/armin-sa/Projects/lm/e7.ckpt"
test_data_path = "/home/is/armin-sa/Projects/lm/test_id.txt"

decoder_model = models.DecoderModel(vocab_size=64000).load_from_checkpoint(
    checkpoint_path, strict=False
)


decoder_trainer = pl.Trainer(
    limit_test_batches=0.05,
    accelerator="gpu",
    devices=1,
)

decoder_trainer.test(
    model=decoder_model,
    dataloaders=torch.utils.data.DataLoader(
        datasets.DecoderDataSet(test_data_path, True)
    ),
)
