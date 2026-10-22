#!/usr/bin/env bash
# Multi-head fine-tuning: same settings as 05_finetune.sh, plus a second head trained
# on the 61 Materials Project structures that contain only Mo and S (replay data),
# so the shared weights are pulled toward what MACE-MP-0 learned from them.
# Multi-head mode needs the foundation model's atomic reference energies (E0s).
# Usage: scripts/09_multihead_finetune.sh [model name] [training file] [seed]
set -e
NAME=${1:-mos2_multihead}
TRAIN=${2:-data/train.xyz}
SEED=${3:-1}
mace_run_train \
  --name "$NAME" \
  --foundation_model small \
  --multiheads_finetuning True \
  --atomic_numbers "[16, 42]" \
  --filter_type_pt combinations \
  --num_samples_pt 61 \
  --force_mh_ft_lr True \
  --train_file "$TRAIN" \
  --valid_file data/valid.xyz \
  --energy_key REF_energy \
  --forces_key REF_forces \
  --E0s foundation \
  --loss ef \
  --energy_weight 1.0 \
  --forces_weight 100.0 \
  --lr 0.005 \
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
