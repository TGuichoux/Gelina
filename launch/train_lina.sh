#!/bin/bash
#SBATCH --job-name=lina-beat-pt_ltts-mls-ggsp-167M
#SBATCH --partition=hard
#SBATCH --exclude=lizzy,thin,led
#SBATCH --time=2-00:00:00  
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --output=out/bash/lina-beat-pt_ltts-mls-ggsp-167M.log
#SBATCH --error=out/bash/lina-beat-pt_ltts-mls-ggsp-167M.err

source activate gelina
python scripts/train/train_lina.py +experiment=lina-beat-pt_ltts-mls-ggsp-167M