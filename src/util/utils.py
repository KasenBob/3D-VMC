import os

import numpy as np
import open3d
import mcubes
import math
import torch
from omegaconf import OmegaConf, DictConfig


def load_config(cfg_path: str) -> DictConfig:
    """
        Load configuration file. `base_config.yaml` is taken as a base for every config.
    :param cfg_path: Path to the configuration file
    :return: Loaded configuration
    """
    base_cfg = OmegaConf.load('config/base_config.yaml')
    curr_cfg = OmegaConf.load(cfg_path)
    return OmegaConf.merge(base_cfg, curr_cfg)