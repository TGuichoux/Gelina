#!/bin/bash
#SBATCH --job-name=sim-r-cosyvoice
#SBATCH --partition=hard,electronic,funky
#SBATCH --time=2-00:00:00  
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --output=out/bash/sim-r-cosyvoice.log
#SBATCH --error=out/bash/sim-r-cosyvoice.err

source activate matchalina-r
python scripts/eval/sim_r.py --gt_dir=out/segmented_outputs_V2/gt_segments/ --res_dir=out/segmented_outputs_V2/cosyvoice/