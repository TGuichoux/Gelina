#!/bin/bash
#SBATCH -A jgk@a100   
#SBATCH --job-name=lina
#SBATCH -C a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=32
#SBATCH --time=20:00:00
#SBATCH --output=out/bash/jz/lina-ltts-mls-ggsp-167M-3a100.out
#SBATCH --error=out/bash/jz/lina-ltts-mls-ggsp-167M-3a100.err

module purge
module load arch/a100
module load pytorch-gpu/py3/2.3.0

torchrun --standalone --nproc_per_node=3 scripts/train_lina.py \
    +experiment=lina-ltts-mls-ggsp-167M \
    trainer.logger.name=lina-ltts-mls-ggsp-167M-3a100 \
    +trainer.logger.offline=True \
    trainer.devices=3 \
    datamodule.num_workers=2 \
    datamodule.token_by_batch=50000 \
    datamodule.save_dir=/lustre/fswork/projects/rech/jgk/usj14pf/ \
    trainer.logger.save_dir=/lustre/fswork/projects/rech/jgk/usj14pf/MATCHALINA/lina \
    trainer.callbacks.0.dirpath=/lustre/fswork/projects/rech/jgk/usj14pf/ \
    +datamodule.tag_source=False
    #model.load_weights='/lustre/fswork/projects/rech/jgk/usj14pf/MATCHALINA/checkpoints/pt_speech/ltts_mls_ggsp_random_100kstep_speech_loss=4.4371.ckpt'