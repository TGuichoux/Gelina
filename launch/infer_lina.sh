#!/bin/bash
#SBATCH --job-name=cloning_lina
#SBATCH --partition=hard,electronic
#SBATCH --time=2-00:00:00  
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --output=out/bash/cloning_infer_lina.out
#SBATCH --error=out/bash/cloning_infer_lina.err

source activate matchalina-r
python scripts/inference/clone_segments_lina.py batch_size=1 mode='cloning'