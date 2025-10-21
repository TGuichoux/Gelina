#!/bin/bash
#SBATCH --job-name=vq
#SBATCH --partition=hard
#SBATCH --time=2-00:00:00  
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --output=out/bash/vq_qz2-d512.log
#SBATCH --error=out/bash/vq_qz2-d512.err

source activate gelina
python scripts/train/train_vq.py +experiment=qz2-d512