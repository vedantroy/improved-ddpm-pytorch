#! /bin/bash
set -o xtrace

FILE_ARGS='--config-file=./config/fixed_variance.yaml --dir-train=~/dataset/train --dir-val=~/dataset/val'
TIME_ARGS='--target-time=8hr --batch-rate=1'
# Original batch size in paper is 128
# Scale learning rate linearly to batch size
SCHEDULE_ARGS='--warmup=500 --lr="2.3*1e-4"'
BATCH_ARGS='--batch-size=302 --micro-batches=1'
METRICS_ARGS='--checkpoints=10 --diffusion-logs=200 --evals=100'

# disable typer's pretty tracebacks
env _TYPER_STANDARD_TRACEBACK=1 python3 train.py $FILE_ARGS $TIME_ARGS $SCHEDULE_ARGS $BATCH_ARGS $METRICS_ARGS
