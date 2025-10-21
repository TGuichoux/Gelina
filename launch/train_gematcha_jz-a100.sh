#!/bin/bash
#SBATCH -A jgk@a100   
#SBATCH --job-name=gematcha
#SBATCH -C a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=32
#SBATCH --time=20:00:00
#SBATCH --output=out/bash/jz/gematcha-32-3a100-stream-8seed-seed_do05.out
#SBATCH --error=out/bash/jz/gematcha-32-3a100-stream-8seed-seed_do05.err

module purge
module load arch/a100
module load pytorch-gpu/py3/2.3.0

name=gematcha-32-3a100-stream-32-8seed-seed_do05
WANDB__SERVICE_WAIT=300 \
torchrun --standalone --nproc_per_node=3 scripts/train_gematcha.py \
    trainer.logger.name="$name" \
    +experiment=streaming_32 \
    +trainer.logger.offline=True \
    trainer.devices=3 \
    datamodule.root_dir=/lustre/fswork/projects/rech/jgk/usj14pf/ \
    trainer.logger.save_dir=/lustre/fswork/projects/rech/jgk/usj14pf/GEMATCHA/ \
    trainer.callbacks.0.dirpath=/lustre/fswork/projects/rech/jgk/usj14pf/GEMATCHA/checkpoints/"$name" \
