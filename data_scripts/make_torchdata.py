from dataclasses import dataclass
import os
import io
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch as th
from tqdm import tqdm
import yahp as hp
import torchvision
import torchdata.datapipes as dp
from torch.utils.data import DataLoader

def load_img(x):
    return torchvision.io.read_image(x)

def save_tensor(t):
    buf = io.BytesIO()
    th.save(t, buf)
    return buf.getvalue()

def build_datapipe(root_dir, batch_size):
    datapipe = dp.iter.FSSpecFileLister(root_dir)
    datapipe = datapipe.map(load_img)
    # Needed to use multiple workers 
    # https://sebastianraschka.com/blog/2022/datapipes.html
    # > First, note that we used a `ShardingFilter` in the previous `build_data_pipe` function:
    # > As of this writing, this is a necessary workaround to avoid data duplication when 
    # > we use more than 1 worker. [...]

    # Don't use the datapipe's batching b/c it conflicts with
    # the dataloader's batching (which will collate unnecessarily)
    datapipe = datapipe.sharding_filter()
    return datapipe

@dataclass
class Args(hp.Hparams):
    in_dir: str = hp.required("input directory")
    out_dir: str = hp.required("output directory")
    batch_size: int = hp.required("# of images per parquet file")

if __name__ == "__main__":
    args = Args.create(None, None, cli_args=True)
    datapipe = build_datapipe(args.in_dir, args.batch_size)

    dl = DataLoader(datapipe, batch_size=args.batch_size, num_workers=8)

    _, _, files = next(os.walk(args.in_dir))
    total_files = len(files)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True)

    for idx, batch in enumerate(tqdm(dl, total=total_files / args.batch_size)):
        assert batch[0].shape == (3, 256, 256)
        df = pa.table({"img": [save_tensor(item) for item in batch]})
        pq.write_table(df, out_dir / f"{idx}.parquet")