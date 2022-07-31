from dataclasses import dataclass
import os
import io
from pathlib import Path
import shutil

import pyarrow as pa
import pyarrow.parquet as pq
import torch as th
from tqdm import tqdm
import yahp as hp
import torchvision
import torchvision.transforms.functional as TVF
import torchdata.datapipes as dp
from torch.utils.data import DataLoader


def load_img(x):
    return torchvision.io.read_image(x)

def resize_img(x):
    return TVF.resize(x, (64, 64))

def save_tensor(t, shape):
    assert t.dtype == th.uint8
    assert tuple(t.shape) == shape
    buf = io.BytesIO()
    th.save(t, buf)
    return buf.getvalue()


def build_datapipe(root_dir, batch_size):
    datapipe = dp.iter.FSSpecFileLister(root_dir)
    datapipe = datapipe.map(load_img)
    datapipe = datapipe.map(resize_img)
    # Needed to use multiple workers
    # https://sebastianraschka.com/blog/2022/datapipes.html
    # > First, note that we used a `ShardingFilter` in the previous `build_data_pipe` function:
    # > As of this writing, this is a necessary workaround to avoid data duplication when
    # > we use more than 1 worker. [...]

    datapipe = datapipe.sharding_filter()
    # There's a seriously weird bug where if you
    # don't use the datapipe's batch but use the DataLoader's batching
    # then the parquet files will be huge (some sort of data duplication issue)
    return datapipe.batch(batch_size)


@dataclass
class Args(hp.Hparams):
    in_dir: str = hp.required("input directory")
    out_dir: str = hp.required("output directory")
    batch_size: int = hp.required("# of images per parquet file")
    overwrite: bool = hp.optional("overwrite existing files", default=False)


if __name__ == "__main__":
    args = Args.create(None, None, cli_args=True)
    datapipe = build_datapipe(args.in_dir, args.batch_size)

    dl = DataLoader(datapipe, batch_size=1, num_workers=8)

    _, _, files = next(os.walk(args.in_dir))
    total_files = len(files)

    if args.overwrite:
        shutil.rmtree(args.out_dir, ignore_errors=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True)

    tensors_saved = 0

    # The total # of batches are incorrect since near the end some batches will
    # be half batches
    for idx, batch in enumerate(tqdm(dl, total=total_files / args.batch_size)):
        # `item[0]` to undo the DataLoader's collating
        tensors = [save_tensor(item[0], shape=(3, 64, 64)) for item in batch]
        tensors_saved += len(tensors)
        df = pa.table({"img": tensors})
        pq.write_table(df, out_dir / f"{idx}.parquet")

    print(f"Saved {tensors_saved} tensors")
