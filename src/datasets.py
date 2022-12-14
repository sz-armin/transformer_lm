import torch, sys, random
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import random_split, DataLoader
from typing import Optional


class EncoderTrainDataSet(torch.utils.data.Dataset):
    def __init__(self, file_path, locs_path, num_workers):
        super().__init__()
        self.num_workers = num_workers
        self.sample_locs = np.load(locs_path)
        self.file_handles = [open(file_path, "rb") for _ in range(num_workers)]
        self.file_handles.append(open(file_path, "rb"))
        self.max_token = 2500

    def __len__(self):
        return len(self.sample_locs)

    def __getitem__(self, idx):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
        else:
            worker_id = worker_info.id
        self.file_handles[worker_id].seek(self.sample_locs[idx])
        count = int.from_bytes(
            self.file_handles[worker_id].read(2), byteorder=sys.byteorder, signed=False
        )
        arr = np.frombuffer(
            self.file_handles[worker_id].read(count * 4), count=count, dtype=np.int32
        )
        return arr


class EncoderDataModule(pl.LightningDataModule):
    def __init__(self, dataset: torch.utils.data.Dataset, train_bsz: int = 4):
        super().__init__()
        self.dataset = dataset
        self.num_workers = self.dataset.num_workers
        self.train_bsz = train_bsz

        self.max_context = 256

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            train_len = int(len(self.dataset) * 0.80)
            self.dataset_train, self.dataset_val = random_split(
                self.dataset,
                [train_len, len(self.dataset) - train_len],
                generator=torch.Generator().manual_seed(42),
            )

        if stage == "test" or stage is None:
            self.dataset_test = self.dataset

        if stage == "predict" or stage is None:
            self.dataset_predict = self.dataset

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.train_bsz,
            collate_fn=self._collate_wrapper,
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=True,
            prefetch_factor=4,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=self.train_bsz,
            collate_fn=self._collate_wrapper,
            num_workers=self.num_workers,
            prefetch_factor=4,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            batch_size=2,
            collate_fn=self._collate_wrapper,
            num_workers=self.num_workers,
            prefetch_factor=4,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.dataset_predict,
            batch_size=2,
            collate_fn=self._collate_wrapper,
            num_workers=self.num_workers,
            prefetch_factor=4,
        )

    def on_after_batch_transfer(self, batch, dataloader_idx):
        batch = batch.unfold(1, min(self.max_context, int(batch.shape[1] / 2)), 1)
        batch = batch[(batch != 3).logical_and(batch != 1).any(axis=2)]

        # TODO batch 1
        # Bucketing
        if batch.shape[0] * batch.shape[1] > 30000:
            batch = batch[
                random.sample(range(batch.shape[0]), k=int(30000 / batch.shape[1]))
            ]

        return batch[:, :-1], batch[:, -1]

    def _collate_wrapper(self, batch):
        # TODO filter large batches
        b_max_len = len(max(batch, key=len))
        if b_max_len <= self.max_context:
            max_pad = 2 * b_max_len
        else:
            max_pad = b_max_len + self.max_context
        batch = np.array(
            [
                np.pad(
                    x,
                    (max_pad - len(x), 0),
                    "constant",
                    constant_values=(3),
                )
                for x in batch
            ]
        )
        # faster right_padding: batch = np.column_stack(list(itertools.zip_longest(*l, fillvalue=3)))
        # TODO type
        return torch.as_tensor(batch, dtype=torch.long)


class DecoderDataSet(torch.utils.data.Dataset):
    def __init__(self, file_path, is_npy=False, all=False):
        super().__init__()
        if is_npy:
            self.data = np.load(file_path)
        else:
            self.data = np.fromfile(file_path, sep=" ", dtype=np.int64)
        self.context = 256 + 1
        self.start_seed = random.randint(0, self.context)
        if all:
            self.start_seed = 0

    def __len__(self):
        return (len(self.data) - self.start_seed) // self.context

    def __getitem__(self, idx):
        start = idx * self.context + self.start_seed
        arr = self.data[start : start + self.context]
        return arr


class DecoderDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: torch.utils.data.Dataset,
        train_bsz: int = 4,
        num_workers: int = 2,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.num_workers = num_workers
        self.train_bsz = train_bsz

    def prepare_data(self):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_bsz,
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=False,
            prefetch_factor=4,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.train_bsz,
            num_workers=self.num_workers,
            persistent_workers=False,
            prefetch_factor=4,
        )

    def on_after_batch_transfer(self, batch, dataloader_idx):
        batch = batch[(batch != 3).all(axis=-1)]

        return batch[:, :-1], batch[:, 1:]


class TestDataSet(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self.data = [
            torch.randint(0, 5, (5,), dtype=torch.int32) for x in range(100000)
        ]
        for x in self.data:
            x[-1] = x[0] + x[1] * x[-2]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
