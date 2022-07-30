import io
import torch as th


def load_tensor(bytes):
    buf = io.BytesIO(bytes)
    return th.load(buf)
