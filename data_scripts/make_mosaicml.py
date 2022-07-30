# I want to just use parquet files in S3
# but 2 things need to be sorted out 1st:
# - https://github.com/pytorch/data/issues/702
# - https://github.com/pytorch/data/issues/703
from dataclasses import dataclass
import os
import io

import torch as th
from tqdm import tqdm
import yahp as hp
import torchvision
import torchdata.datapipes as dp
from torch.utils.data import DataLoader
from composer.datasets.streaming import StreamingDatasetWriter


@dataclass
class Args(hp.Hparams):
    in_dir: str = hp.required("input directory")
    out_dir: str = hp.required("output directory")


def load_img(x):
    return torchvision.io.read_image(x)


def build_datapipes(root_dir):
    datapipe = dp.iter.FSSpecFileLister(root_dir)
    # functional form of `dp.iter.Mapper`
    datapipe = datapipe.map(load_img)
    return datapipe
    # return datapipe.batch(batch_size)


def save_tensor(t):
    buf = io.BytesIO()
    th.save(t, buf)
    return buf.getvalue()


if __name__ == "__main__":
    args = Args.create(None, None, cli_args=True)
    datapipe = build_datapipes(args.in_dir)

    # num_workers > 1 causes infinite loop
    # https://github.com/pytorch/data/issues/702
    # This is also the reason
    dl = DataLoader(datapipe, batch_size=1, num_workers=1)

    _, _, files = next(os.walk(args.in_dir))
    total_files = len(files)

    FIELDS = ["img"]
    with StreamingDatasetWriter(args.out_dir, FIELDS) as writer:
        for batch in tqdm(dl, total=total_files):
            img = batch[0]
            assert img.shape == (3, 256, 256)
            img = save_tensor(img)
            writer.write_sample({"img": img})
