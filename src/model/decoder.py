import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
import pytorch_lightning as pl
import torchvision.models as models

from src.model.attention.layers import (
    AttentionLayers
)

from src.model.losses import (
    DiceLoss, 
    CEDiceLoss, 
    FocalLoss
)


class Contrastiveblock(pl.LightningModule):
    def __init__(self, n_features):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(64, 64, bias=False),
            nn.ReLU(),
            nn.Linear(64, 128, bias=False),
        )
    def forward(self, x):
        #print(self.backbone.fc)
        return self.projector(x)


class ResBlock(pl.LightningModule):
    def __init__(self, chan):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(chan, chan, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(chan, chan, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(chan, chan, 1)
        )

    def forward(self, x):
        return self.net(x) + x


class VoxelDecoder(pl.LightningModule):
    def __init__(
            self,
            patch_num: int = 4,
            num_cnn_layers: int = 3,
            num_resnet_blocks: int = 2,
            cnn_hidden_dim: int = 64,
            voxel_size: int = 32,
            dim: int = 512,
            depth: int = 6,
            heads: int = 8,
            dim_head: int = 64,
            attn_dropout: float = 0.0,
            ff_dropout: float = 0.0,
    ):
        super().__init__()
        if voxel_size % patch_num != 0:
            raise ValueError('voxel_size must be dividable by patch_num')
        self.patch_num = patch_num
        self.voxel_size = voxel_size
        self.patch_size = voxel_size // patch_num
        self.emb = nn.Embedding(patch_num ** 3, dim)
        self.transformer = AttentionLayers(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            causal=False,
            cross_attend=True
        )
        self.layer_norm = nn.LayerNorm(dim)
        self.to_patch = nn.Linear(dim, self.patch_size ** 3)

    def forward(self, context, context_mask: Tensor = None):
        x = self.emb(torch.arange(self.patch_num ** 3, device=context.device))
        x = x.unsqueeze(0).repeat(context.shape[0], 1, 1)
        out = self.transformer(x=x, context=context, context_mask=context_mask)
        out = self.layer_norm(out)
        out = rearrange(out, 'b (h w c) d -> b d h w c', h=self.patch_num, w=self.patch_num, c=self.patch_num)
        # out -> [32, 192, 4, 4, 4]
        #out = self.decoder(out)
        #print(out.shape)
        return out


class VoxelCNN(pl.LightningModule):
    def __init__(
        self,
        num_resnet_blocks: int = 2,
        cnn_hidden_dim: int = 64,
        num_cnn_layers: int = 3,
        dim: int = 512
    ):
        super().__init__()
        has_resblocks = num_resnet_blocks > 0
        dec_chans = [cnn_hidden_dim] * num_cnn_layers
        dec_init_chan = dim if not has_resblocks else dec_chans[0]
        dec_chans = [dec_init_chan, *dec_chans]
        dec_chans_io = list(zip(dec_chans[:-1], dec_chans[1:]))
        dec_layers = []
        for (dec_in, dec_out) in dec_chans_io:
            dec_layers.append(nn.Sequential(nn.ConvTranspose3d(dec_in, dec_out, 4, stride=2, padding=1), nn.ReLU()))
        for _ in range(num_resnet_blocks):
            dec_layers.insert(0, ResBlock(dec_chans[1]))
        if num_resnet_blocks > 0:
            dec_layers.insert(0, nn.Conv3d(dim, dec_chans[1], 1))
        dec_layers.append(nn.Conv3d(dec_chans[-1], 1, 1))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        return self.decoder(x)


class VoxelMLP(pl.LightningModule):
    def __init__(self, dim_model, output_resolution=32):
        super().__init__()
        # Define output layers
        self.linear_z = nn.Linear(dim_model, output_resolution)
        self.linear_y = nn.Linear(dim_model, output_resolution)
        self.linear_x = nn.Linear(dim_model, output_resolution)
        # Initialize output layers
        torch.nn.init.xavier_uniform_(self.linear_z.weight)
        torch.nn.init.xavier_uniform_(self.linear_y.weight)
        torch.nn.init.xavier_uniform_(self.linear_x.weight)

    def forward(self, x):
        z_factors = self.linear_z(x).sigmoid()
        y_factors = self.linear_y(x).sigmoid()
        x_factors = self.linear_x(x).sigmoid()

        return z_factors, y_factors, x_factors
