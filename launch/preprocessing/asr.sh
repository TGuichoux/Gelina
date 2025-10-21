#!/bin/bash
#SBATCH --job-name=asr
#SBATCH --partition=hard
#SBATCH --exclude=thin
#SBATCH --time=2-00:00:00  
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --output=out/bash/asr_preprocessing.log
#SBATCH --error=out/bash/asr_preprocessing.err

source activate preprocess-data
python scripts/preprocessing/asr.py