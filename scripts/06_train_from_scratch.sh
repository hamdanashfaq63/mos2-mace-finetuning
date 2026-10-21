#!/usr/bin/env bash
# Baseline: train a MACE model of similar size from scratch on the same data,
# to measure how much the pretrained foundation model helps.
# Usage: scripts/06_train_from_scratch.sh [model name] [training file]
set -e
NAME=${1:-mos2_from_scratch}
TRAIN=${2:-data/train.xyz}
mace_run_train \
  --name "$NAME" \
  --model MACE \
  --hidden_irreps "128x0e" \
  --r_max 6.0 \
  --max_L 0 \
  --correlation 3 \
  --train_file "$TRAIN" \
  --valid_file data/valid.xyz \
  --energy_key REF_energy \
  --forces_key REF_forces \
  --E0s average \
  --loss ef \
  --energy_weight 1.0 \
  --forces_weight 100.0 \
  --lr 0.01 \
  --batch_size 4 \
  --max_num_epochs 300 \
  --ema --ema_decay 0.99 \
  --default_dtype float64 \
  --device cpu \
  --seed 1 \
  --model_dir models \
  --log_dir results/train_logs \
  --checkpoints_dir results/checkpoints \
  --results_dir results/train_logs
