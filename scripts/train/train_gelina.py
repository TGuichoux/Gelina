
# scripts/train_lina.py
import os

import torch
import torch.multiprocessing as mp

mp.set_start_method("spawn", force=True) 
os.environ["OMP_NUM_THREADS"] = "1"            
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["ARROW_PRE_0_15_IPC_FORMAT"] = "1"  
os.environ["WANDB_SILENT"] = "true"
os.environ["WANDB_DISABLE_SERVICE"] = "true"

# cache Triton kernels
# os.environ["TRITON_CACHE_DIR"] = os.path.expanduser("~/.triton/cache")
os.environ["HYDRA_FULL_ERROR"] = "0"
os.environ["GRAD_CKPT"] = "1"

import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from pytorch_lightning.callbacks import ModelSummary

import logging
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning.utilities.parsing").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning.trainer.connectors.logger_connector.result").setLevel(logging.ERROR)

import datasets
datasets.utils.logging.set_verbosity_error()

OmegaConf.register_new_resolver(
    "cpu_minus1",                                      
    lambda: max(1, (os.cpu_count() or 1) - 1)       
)

@hydra.main(config_path="../../configs/gelina", config_name="default", version_base=None)
def main(cfg: DictConfig):
    
    pl.seed_everything(cfg.seed_everything)

    # link datamodule → model field (Hydra-style)
    if "quant_layer_speech" not in cfg.model or cfg.model.quant_layer_speech is None:
        print(cfg.datamodule)
        cfg.model.quant_layer_speech = cfg.datamodule.quant_layer_speech

    if "quant_layer_motion" not in cfg.model or cfg.model.quant_layer_motion is None:

        cfg.model.quant_layer_motion = cfg.datamodule.quant_layer_motion

    datamodule = instantiate(cfg.datamodule)
    print("✔ Loaded datamodule:")
    print(OmegaConf.to_yaml(cfg.datamodule), "\n")

    model = instantiate(cfg.model)
    print("✔ Instantiated model")

    trainer = instantiate(cfg.trainer, callbacks=[ModelSummary(max_depth=-1)])
    
 
    print("Number of devices:", trainer.num_devices)
    print("✔ Instantiated Trainer")

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()

# example usage:
# python scripts/train_gelina.py trainer.logger=False 'trainer.callbacks=[]'