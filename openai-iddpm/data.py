import io

import pyarrow.parquet as pq
import torch as th
import torchdata.datapipes as dp
import pyarrow.parquet as pq

def load_tensor(bytes):
    buf = io.BytesIO(bytes)
    return th.load(buf)

def identity(x):
    return x

def load_parquet(path):
    table = pq.read_table(path)
    imgs = table["img"]
    imgs = [load_tensor(img.as_py()) for img in imgs]
    return imgs


def infinite_dataloader(batch_size, image_size, dir):
    def assert_dims(x):
        assert x.shape == (3, image_size, image_size)
        return x

    datapipe = dp.iter.FSSpecFileLister(dir)
    datapipe = datapipe.map(load_parquet)
    datapipe = datapipe.flatmap(identity)
    datapipe = datapipe.map(assert_dims)
    datapipe = datapipe.shuffle()
    datapipe = datapipe.cycle()
    datapipe = datapipe.batch(batch_size)
    return iter(datapipe)