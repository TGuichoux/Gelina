#!/bin/bash
#SBATCH --job-name=cfm
#SBATCH --partition=hard
#SBATCH --exclude=thin
#SBATCH --time=2-00:00:00  
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --output=out/bash/cfm-test.log
#SBATCH --error=out/bash/cfm-test.err

source activate gelina
python scripts/train/train_cfm.py +experiment=rec_loss