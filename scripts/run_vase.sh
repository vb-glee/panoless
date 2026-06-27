#!/bin/bash
set -e

CUDA_VISIBLE_DEVICES=0  # <-- change to a free GPU index

DATA=/home/ad158/gs-dataset/custom/vase_t45-120_p0-60_plus_center
LOG=/home/ad158/panoless/logs/vase_$(date +%Y%m%d_%H%M%S).log

mkdir -p /home/ad158/panoless/logs

cd /home/ad158/panoless
source /home/ad158/miniconda/etc/profile.d/conda.sh
conda activate panoless

export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
TORCH_LIB=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$TORCH_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

nohup python train.py \
    -s "$DATA" \
    --eval \
    --albedo_bias 0 \
    >& "$LOG" &

echo "Launched PID $! — tail -f $LOG"
