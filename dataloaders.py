import pyarrow as pa
import pyarrow.parquet as pq
import torch as th
from tqdm import tqdm
import yahp as hp
import torchdata.datapipes as dp
from torch.utils.data import DataLoader
import pyarrow.parquet as pq

from utils import load_tensor


def identity(x):
    return x


def print_identity(x):
    print(x)
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
    #datapipe = datapipe.cycle(1_000)
    datapipe = datapipe.cycle(1_000)
    # datapipe = datapipe.cycle()
    datapipe = datapipe.sharding_filter()
    # We want no parallelism b/c this might lead to batches
    # that are smaller than batch_size, which will break grad_accum
    return DataLoader(datapipe, batch_size=batch_size, num_workers=1)
