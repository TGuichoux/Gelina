
# scripts/train_lina.py
import os

# cache Triton kernels
os.environ["TRITON_CACHE_DIR"] = os.path.expanduser("~/.triton/cache")
os.environ["HYDRA_FULL_ERROR"] = "1"

import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate


@hydra.main(config_path="../../configs/lina", config_name="default")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed_everything)

    # link datamodule → model field (Hydra-style)
    if "quant_layer" not in cfg.model or cfg.model.quant_layer is None:
        print(cfg.datamodule)
        cfg.model.quant_layer = cfg.datamodule.quant_layer_speech

    datamodule = instantiate(cfg.datamodule)
    print("✔ Loaded datamodule:")
    print(OmegaConf.to_yaml(cfg.datamodule), "\n")

    model = instantiate(cfg.model)
    print("✔ Instantiated model")

    trainer = instantiate(cfg.trainer)
    print("Accelerator:", trainer.accelerator)
    print("Number of devices:", trainer.num_devices)
    print("✔ Instantiated Trainer")

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()

# example usage:
# python scripts/train/train_lina.py trainer.logger=False 'trainer.callbacks=[]'