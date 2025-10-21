#!/bin/bash
#SBATCH --job-name=generate_latent
#SBATCH --partition=hard
#SBATCH --time=2-00:00:00  
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --output=out/bash/generate_latents.log
#SBATCH --error=out/bash/generate_latents.err

source activate gelina
python scripts/preprocessing/generate_latent_dataset.py batch_size=1