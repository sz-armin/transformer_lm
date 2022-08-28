import torch, sys, glob, os, math
from pathlib import Path
import numpy as np
import torch.multiprocessing as mp
mp.set_start_method('fork')
from multiprocessing import Pool
from copy import deepcopy

sample_locs = [0]
with open("data/ru_small_id.txt", "r") as f_in:
    line = f_in.readline()
    while line:
        f_in.tell() #returns the location of the next line
        line = f_in.readline()
        sample_locs.append(f_in.tell())


class CC100DataSet(torch.utils.data.Dataset):
    def __init__(self, file_path, num_workers):
        super().__init__()
        self.path=file_path
        self.sample_locs = deepcopy(sample_locs)
        self.file_handles = [open(file_path, "r") for _ in range(num_workers)]
        self.file_handles.append(open(file_path, "r"))

    def __len__(self):
        return len(self.sample_locs)

    def __getitem__(self, idx):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
        else:
            worker_id = worker_info.id
        # self.file_handles[worker_id].seek(self.sample_locs[idx])
        
        line = linecache.getline(self.path, idx)
        # arr = np.fromstring(line, dtype=np.int32, sep=" ")
        if len(line) > 0:
            arr = torch.tensor(list(map(int, line.split(' '))), dtype=torch.int16)
            return line
        # return np.copy(arr)
        return '0'

def collate_wrapper(batch):
    # print(batch)
    return list(map(lambda x: torch.tensor(list(map(int, x.split(' '))), dtype=torch.int16), batch))
    # return torch.utils.data.default_convert(batch)


# num_workers = 8
# ds = CC100DataSet("data/ru_small_id.txt", num_workers)
# data_loader = torch.utils.data.DataLoader(
#     ds,
#     num_workers=num_workers,
#     shuffle=False,
#     batch_size=64,
#     collate_fn=collate_wrapper,
#     # prefetch_factor=1,
#     # persistent_workers=True,
#     # pin_memory=True,
# )

# import time
# c = 0
# # l = [0]
# for _ in data_loader:
#     # l[0]=_
#     c+=1

# print(c)

if __name__ == '__main__':
    sample_locs = np.array(sample_locs[:-1])
    import linecache
    num_workers = 4
    ds = CC100DataSet("data/ru_small_id.txt", num_workers)
    data_loader = torch.utils.data.DataLoader(
        ds,
        num_workers=num_workers,
        shuffle=False,
        batch_size=64,
        collate_fn=collate_wrapper,
        # prefetch_factor=1,
        # persistent_workers=True,
        # pin_memory=True,
    )
    # import time
    c = 0
    # l = [0]
    for _ in data_loader:
        # l[0]=_
        c+=1

    print(c)