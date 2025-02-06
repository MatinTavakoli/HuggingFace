#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --mem=10GB
#SBATCH --time=6:00:00

python -c 'import torch; print(torch.cuda.is_available())'
