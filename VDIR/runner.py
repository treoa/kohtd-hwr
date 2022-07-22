import os
import torch
import hydra

from omegaconf import OmegaConf, DictConfig


@hydra.main(config_path='conf', config_name='config')
def run_model(cfg: DictConfig) -> None:
    os.makedirs('logs', exist_ok=True)
    print(OmegaConf.to_yaml(cfg))
