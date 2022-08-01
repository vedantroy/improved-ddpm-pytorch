from types import SimpleNamespace
import typing

import torch as th
import wandb
import torchvision.transforms.functional as TVF
from composer.utils import ensure_tuple
from composer import Callback
from composer.core import State, Time
from composer.loggers import Logger, WandBLogger, LogLevel


def normalized_to_bytes(img):
    return (((img + 1) / 2) * 255).to(th.uint8)


class DiffusionMonitor(Callback):
    def __init__(self, interval: str):
        self.interval = Time.from_timestring(interval)
        # wandb bug: https://github.com/wandb/wandb/issues/4027
        # self.to_display = to_display
        # self.key = key

    def before_loss(self, state: State, logger: Logger) -> None:
        if state.timestamp.get(self.interval.unit).value % self.interval.value == 0:
            outputs = typing.cast(SimpleNamespace, state.outputs)
            assert isinstance(
                outputs, SimpleNamespace
            ), f"Invalid output type: {type(outputs)}"

            x_0 = state.batch
            x_t, t, x_0_pred = outputs.x_t, outputs.t, outputs.x_0_pred

            # x_0 = x_0[: self.to_display]
            # x_t = x_t[: self.to_display]
            # t = t[: self.to_display]
            # assert x_0.shape == x_t.shape and x_t.shape[0] == t.shape[0]

            x_0 = x_0[0]
            x_t = x_t[0]
            t = t[0]
            x_0_pred = x_0_pred[0]

            assert x_0.dtype == th.uint8
            # TVF accepts byte tensors not [-1, 1] tensors
            # x_t = (((x_t + 1) / 2) * 255).to(th.uint8)
            x_t = normalized_to_bytes(x_t)
            x_0_pred = normalized_to_bytes(x_0_pred)
            x_0_pil = TVF.to_pil_image(x_0)
            x_t_pil = TVF.to_pil_image(x_t)
            x_0_pred_pil = TVF.to_pil_image(x_0_pred)

            # key = self.key
            # cur_batch = state.timestamp.get(self.interval.unit).value
            # key = f"{key}_{cur_batch}"
            # key = self.key
            # table = make_table(x_0=x_0, x_t=x_t, t=t)
            for destination in ensure_tuple(logger.destinations):
                if isinstance(destination, WandBLogger):
                    destination.log_data(
                        state,
                        LogLevel.BATCH,
                        {
                            "prediction": {
                                "x_0": wandb.Image(x_0_pil),
                                "x_t": wandb.Image(x_t_pil),
                                "x_0_pred": wandb.Image(x_0_pred_pil),
                                "t": t,
                            }
                        },
                    )
                    # print("RUNNING LOG")
                    # print(state.timestamp.get(self.interval.unit).value)
                    # destination.log_data(state, LogLevel.BATCH, {key: table})


def make_table(*, x_0, x_t, t):
    table = wandb.Table(columns=["x_0", "x_t", "t"])
    for x_0_, x_t_, t_ in zip(x_0, x_t, t):
        x_0_pil = TVF.to_pil_image(x_0_)
        x_t_pil = TVF.to_pil_image(x_t_)
        table.add_data(wandb.Image(x_0_pil), wandb.Image(x_t_pil), t_.item())
    return table
