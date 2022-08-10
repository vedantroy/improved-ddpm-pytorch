#! /bin/bash
set -o xtrace

FILE_ARGS='--config-file=./config_model/fixed_variance.yaml --dir-train=~/dataset/train --dir-val=~/dataset/val'
TIME_ARGS='--target-time=10hr --batch-rate=2.34'
# Original batch size in paper is 128
# Scale learning rate by sqrt to batch size
SCHEDULE_ARGS='--warmup=500 --lr="1e-4"'
BATCH_ARGS='--batch-size=16 --micro-batches=1 --precision=fp32'
METRICS_ARGS='--checkpoints=12 --diffusion-logs=200 --evals=14000'

# disable typer's pretty tracebacks
env _TYPER_STANDARD_TRACEBACK=1 python3 train.py $FILE_ARGS $TIME_ARGS $SCHEDULE_ARGS $BATCH_ARGS $METRICS_ARGS
