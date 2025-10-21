#!/bin/bash
#SBATCH --job-name=without_pretrain
#SBATCH --partition=hard,electronic
#SBATCH --time=2-00:00:00  
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --output=out/bash/without_pretrain_infer.out
#SBATCH --error=out/bash/without_pretrain_infer.err

source activate matchalina-r
python scripts/inference/clone_segments_cfm.py batch_size=1 mode='without_pretrain'