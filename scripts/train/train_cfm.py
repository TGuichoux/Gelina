import os
# Enable full Hydra errors if you like
os.environ["HYDRA_FULL_ERROR"] = "1"

import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from pytorch_lightning.callbacks import ModelSummary

@hydra.main(config_path="../../configs/cfm", config_name="default", version_base=None)
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed_everything)

    dm = instantiate(cfg.datamodule)

    print("✔ Loaded datamodule:")
    print(OmegaConf.to_yaml(cfg.datamodule), "\n")

    model = instantiate(cfg.model)
    print("✔ Instantiated model:")

    trainer = instantiate(cfg.trainer, callbacks=[ModelSummary(max_depth=-1)])

    print("Number of devices:", trainer.num_devices)
    print(f"✔ Instantiated Trainer.")

    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    main()