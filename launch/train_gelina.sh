#!/bin/bash
#SBATCH --job-name=gelina-test
#SBATCH --partition=hard
#SBATCH --exclude=lizzy,thin,led
#SBATCH --time=2-00:00:00  
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --output=out/bash/gelina-test.log
#SBATCH --error=out/bash/gelina-test.err

source activate gelina
python scripts/train/train_gelina.py +experiment=beat-pt_ltts_mls_ggsp_random_motion-167M-1qz-lr5e-5 datamodule.num_workers=8