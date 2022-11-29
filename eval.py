import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision import transforms
import pytorch_lightning as pl
import argparse
from pytorch_lightning.loggers import TensorBoardLogger

from src.util import load_config
from src.data import ShapeNetDataModule
from src.model import transformer
from src.utils import get_mlflow_tags

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, default='config/base_model.yaml')
parser.add_argument("--experiment_name", type=str, default='transformer')
parser.add_argument("--ckpt_path", type=str, default='')
args = parser.parse_args()

def main():
# =================================================================================
    # Load config file
    cfg = load_config(args.config_path)
# =================================================================================
    data_module = ShapeNetDataModule(cfg.data)
    model = transformer(cfg, restart=True)

    model = model.load_from_checkpoint(args.ckpt_path, cfg=cfg, restart=True)
    print("Have loaded!")

    for p in model.parameters():
        p.requires_grad = False
# =================================================================================
    logger = TensorBoardLogger("test_logs/", name="transformer")
# =================================================================================
    trainer = pl.Trainer(callbacks=None, logger=logger, **cfg.trainer)
    #trainer = pl.Trainer(callbacks=None, logger=False, **cfg.trainer)
    trainer.test(model, data_module.test_dataloader())
# =================================================================================

if __name__ == '__main__':
    main()




