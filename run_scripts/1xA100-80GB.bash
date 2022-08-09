#! /usr/bin/bash
set -o xtrace

FILE_ARGS='--config-file=./config/fixed_variance.yaml --dir-train=~/dataset/train --dir-val=~/dataset/val'
TIME_ARGS='--target-time=4hr --batch-rate=5.1'
SCHEDULE_ARGS='--warmup=100 --lr=1e-4'
BATCH_ARGS='--batch-size=302 --micro-batches=1'
METRICS_ARGS='--checkpoints=100 --diffusion-logs=100 --evals=100'

# disable typer's pretty tracebacks
env WANDB_MODE="disabled" _TYPER_STANDARD_TRACEBACK=1 python3 train.py $FILE_ARGS $TIME_ARGS $SCHEDULE_ARGS $BATCH_ARGS $METRICS_ARGS
