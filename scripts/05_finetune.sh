#!/usr/bin/env bash
# Fine-tune MACE-MP-0 small on the MoS2 DFT data.
# Usage: scripts/05_finetune.sh [model name] [training file] [seed] [learning rate]
set -e
NAME=${1:-mos2_finetuned}
TRAIN=${2:-data/train.xyz}
SEED=${3:-1}
LR=${4:-0.005}
mace_run_train \
  --name "$NAME" \
  --foundation_model small \
  --multiheads_finetuning False \
  --train_file "$TRAIN" \
  --valid_file data/valid.xyz \
  --energy_key REF_energy \
  --forces_key REF_forces \
  --E0s average \
  --loss ef \
  --energy_weight 1.0 \
  --forces_weight 100.0 \
  --lr "$LR" \
  --batch_size 4 \
  --max_num_epochs 150 \
  --ema --ema_decay 0.99 \
  --default_dtype float64 \
  --device cpu \
  --seed "$SEED" \
  --model_dir models \
  --log_dir results/train_logs \
  --checkpoints_dir results/checkpoints \
  --results_dir results/train_logs
