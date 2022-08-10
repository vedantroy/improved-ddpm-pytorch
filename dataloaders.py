from pathlib import Path

import pyarrow.parquet as pq
import torchdata.datapipes as dp
from torch.utils.data import DataLoader
import pyarrow.parquet as pq

from utils import load_tensor


def identity(x):
    return x


def load_parquet(path):
    table = pq.read_table(path)
    imgs = table["img"]
    imgs = [load_tensor(img.as_py()) for img in imgs]
    return imgs


def overfit_dataloader(num_batches, batch_size, dir):
    # Notes
    # 1. This dataloader only works w/ parquet files
    # on the local file system
    # 2. We implicitly rely on
    # FSSpecFileLister to be deterministic
    # 3. Using print(item[indexes]) inside of the dataloader
    # loop, we see the same samples are printed everytime
    datapipe = dp.iter.FSSpecFileLister(dir)
    datapipe = datapipe.map(load_parquet)
    datapipe = datapipe.flatmap(identity)
    datapipe = datapipe.header(num_batches * batch_size)
    datapipe = datapipe.shuffle()
    # datapipe = datapipe.cycle(1_000)
    datapipe = datapipe.cycle(1_000)
    # datapipe = datapipe.cycle()
    datapipe = datapipe.sharding_filter()
    # We want no parallelism b/c this might lead to batches
    # that are smaller than batch_size, which will break grad_accum
    return DataLoader(datapipe, batch_size=batch_size, num_workers=1)


def count_files(dir):
    p = Path(dir)
    assert p.is_dir(), f"{dir} is not a directory"
    return sum(1 for _ in p.glob("*"))

def dataloader(dir, batch_size, workers):
    # total_files = count_files(dir)
    # print(f"Total files: {total_files}")
    datapipe = dp.iter.FSSpecFileLister(dir)
    datapipe = datapipe.map(load_parquet)
    datapipe = datapipe.flatmap(identity)
    datapipe = datapipe.shuffle()
    datapipe = datapipe.sharding_filter()
    dl = DataLoader(datapipe, batch_size=batch_size, num_workers=workers, drop_last=True)
    has_item = any(True for _ in dl)
    assert has_item, f"{dir} with {workers} workers yielded no batches"
    return dl


# def train_val_loaders(dir, batch_size, val_batches, num_workers=8):
#    datapipe = dp.iter.FSSpecFileLister(dir)
#    datapipe = datapipe.map(load_parquet)
#    datapipe = datapipe.flatmap(identity)
#    val_samples = val_batches * batch_size
#    train_dp, val_dp = datapipe.fork(num_instances=2, buffer_size=val_samples)
#    val_dp = val_dp.header(val_samples)
#    # TODO: Filter out first N samples since they're used in validation
#    train_dp = train_dp.filter()
#    dl_args = dict(
#        batch_size=batch_size,
#        num_workers=num_workers,
#        drop_last=True,
#    )
#    return SimpleNamespace(
#        train=DataLoader(train_dp.shuffle().sharding_filter(), **dl_args),
#        val=DataLoader(val_dp.sharding_filter(), **dl_args),
#    )
