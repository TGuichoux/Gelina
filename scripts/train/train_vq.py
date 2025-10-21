import os
os.environ["HYDRA_FULL_ERROR"] = "1"

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from hydra.utils import instantiate


@hydra.main(config_path="../../configs/vq", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed_everything)

    datamodule = instantiate(cfg.datamodule)
    model = instantiate(cfg.model_trainer)

    trainer = instantiate(cfg.trainer)

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
