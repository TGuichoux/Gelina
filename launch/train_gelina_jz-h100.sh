#!/bin/bash
#SBATCH -A jgk@h100   
#SBATCH --job-name=lina
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=20:00:00
#SBATCH --output=out/bash/jz/beat-pt_ltts_mls_ggsp_random_motion-167M-1qz-lr5e-5.out
#SBATCH --error=out/bash/jz/beat-pt_ltts_mls_ggsp_random_motion-167M-1qz-lr5e-5.err

module purge
module load arch/h100
module load pytorch-gpu/py3/2.3.1

WANDB__SERVICE_WAIT=300 \
torchrun --standalone --nproc_per_node=1 scripts/train_gelina.py \
    +experiment=beat-pt_ltts_mls_ggsp_random_motion-167M-1qz-lr5e-5 \
    trainer.logger.name=beat-pt_ltts_mls_ggsp_random_motion-167M-1qz-lr5e-5 \
    +trainer.logger.offline=True \
    trainer.devices=1 \
    datamodule.num_workers=8 \
    datamodule.save_dir=/lustre/fswork/projects/rech/jgk/usj14pf/ \
    trainer.logger.save_dir=/lustre/fswork/projects/rech/jgk/usj14pf/MATCHALINA \
    trainer.callbacks.0.dirpath=/lustre/fswork/projects/rech/jgk/usj14pf/ \
    model.load_weights=/lustre/fswork/projects/rech/jgk/usj14pf/MATCHALINA/checkpoints/pt_speech/ltts_mls_ggsp_random_100kstep_speech_loss4.4371.ckpt 