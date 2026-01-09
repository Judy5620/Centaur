#!/bin/bash

#SBATCH -J CENTaUR2
#SBATCH -p gpu_p
#SBATCH --qos gpu_long
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH -t 96:00:00
#SBATCH --constraint=a100_80gb
#SBATCH --nice=10000
#SBATCH --cpus-per-task=20

source activate unsloth_env2

python finetune.py \
--seed 100 \
--model_name_or_path "unsloth/Meta-Llama-3.1-70B-bnb-4bit" \
--max_seq_len 32768 \
--num_train_epochs 5 \
--log_level "info" \
--logging_strategy "steps" \
--logging_steps 1 \
--evaluation_strategy "steps" \
--eval_steps 999999 \
--save_strategy "steps" \
--save_steps 100 \
--learning_rate 5e-5 \
--optim "adamw_8bit" \
--lr_scheduler_type "cosine" \
--weight_decay 0.01 \
--warmup_steps 100 \
--output_dir "centaur2-final-llama" \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 32
