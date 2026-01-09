#!/bin/bash

#SBATCH -J CENTaUR2
#SBATCH -p gpu_p
#SBATCH --qos gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH -t 48:00:00
#SBATCH --constraint=a100_80gb
#SBATCH --nice=10000
#SBATCH --cpus-per-task=20

source activate unsloth_env2

cd ..
python test_adapter.py --model marcelbinz/Llama-3.1-Centaur-8B-adapter
python test_adapter_custom_metrics.py --model marcelbinz/Llama-3.1-Centaur-8B-adapter

cd generalization/
python generalization.py --model marcelbinz/Llama-3.1-Centaur-8B-adapter
python generalization_custom_metrics.py --model marcelbinz/Llama-3.1-Centaur-8B-adapter

cd ..
python merge.py --model marcelbinz-Llama-3.1-Centaur-8B-adapter
