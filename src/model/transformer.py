import os
from time import process_time_ns
from turtle import forward
import numpy as np
from numpy import deprecate_with_doc
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
from einops import rearrange
import torch.nn.functional as F
import torch
from torchmetrics import F1Score

from src.model import EncoderModel, VoxelDecoder, VoxelCNN, Contrastiveblock
from src.model.metrics import compute_iou_score, compute_depth_score, compute_iou
from src.model.losses import DiceLoss, ContrastiveLoss
from src.model.utils import plot_voxels, denormalize
from src.util.voxel2obj import voxel2obj
from src.model.reply_memory import ReplayMemory_Per, Transition, taxonomy_info


class transformer(pl.LightningModule):
    def __init__(self, cfg, restart=False):
        super().__init__()
        self.save_hyperparameters()
        
        self.encoder = EncoderModel(
            attn_dropout=cfg.network.encoder.dropout, 
            model=cfg.network.encoder.model, 
            model_weight=cfg.pretrain,
            pretrained=cfg.network.encoder.pretrained, 
        )
        # freeze pretrain encoder
        if restart == False:
            for p in self.parameters():
                p.requires_grad = False

        self.decoder = VoxelDecoder(
            attn_dropout=cfg.network.decoder.dropout,
            depth=cfg.network.decoder.depth,
            heads=cfg.network.decoder.heads,
            dim=cfg.network.decoder.dim,
            patch_num=cfg.network.decoder.patch_num,
            voxel_size=cfg.network.decoder.voxel_size,
            num_resnet_blocks=cfg.network.decoder.resnet_blocks,
            cnn_hidden_dim=cfg.network.decoder.cnn_hidden_dim,
            num_cnn_layers=cfg.network.decoder.cnn_layers,
        )

        self.cnn = VoxelCNN(
            num_resnet_blocks=cfg.network.decoder.resnet_blocks,
            cnn_hidden_dim=cfg.network.decoder.cnn_hidden_dim,
            num_cnn_layers=cfg.network.decoder.cnn_layers,
            dim=cfg.network.decoder.dim
        )

        self.Contrastiveblock = Contrastiveblock(cfg.network.Contrastiveblock)
        self.Contraloss = ContrastiveLoss(batch_size=cfg.network.Contrastivebs)

        #self.memory = ReplayMemory(1000)
        self.memory = ReplayMemory_Per(1000)
        self.tax_dict = taxonomy_info()

        self.cfg = cfg
        self.threshold = 0.5
        self.sample_batch_num = self.cfg.network.sample_batch_num

    def _polarize(self, data):
        data[data > self.threshold] = 1
        data[data <= self.threshold] = 0
        return data

    def forward(self, image):
        if len(image.size()) == 4:
            context = self.encoder(image)
        else:
            batch_size, view_count = image.size(0), image.size(1)
            image = rearrange(image, 'b v c h w -> (b v) c h w')
            context = self.encoder(image)
            context = rearrange(context, '(b v) l d -> b v l d', b=batch_size, v=view_count)
            context = context.mean(dim=1)
        # context => [1, 197, 768]
        context = self.decoder(context=context).squeeze()
        # context => [1, 32, 32, 32]
        if len(context.size()) == 3:
            context = torch.unsqueeze(context, 0)
        return context
   
    def training_step(self, batch, batch_idx):
        image, aug_image, gt_volume, tax, sample = batch  
        # Replay Memory
        buff_size = self.cfg.network.Contrastivebs - self.cfg.data.loader.train_batch_size
        if self.memory.size() >= buff_size:
            _, transitions = self.memory.sample(buff_size)
            for bs in transitions:
                image = torch.cat((bs.images, image), dim=0)
                aug_image = torch.cat((bs.aug_images, aug_image), dim=0)
                gt_volume = torch.cat((bs.volume, gt_volume), dim=0)
                tax.append(bs.taxonomy_name)
                sample.append(bs.sample_name)
        else:
            image = torch.cat((image, image[:buff_size]), dim=0)
            aug_image = torch.cat((aug_image, aug_image[:buff_size]), dim=0)
            gt_volume = torch.cat((gt_volume, gt_volume[:buff_size]), dim=0)
            tax += tax[:2]
            sample += sample[:2] 
        
        bs = image.size()[0]
        # image.shape => [B, V, 3, H (224), W (224)]
        # gt_volume.shape => [B, 32, 32, 32]
        # context -> [32, 192, 4, 4, 4]
        in_context = torch.cat((image, aug_image), dim=0).squeeze()
        context = self.forward(in_context)
        pre, aug = context.split(bs, 0)
        down_stream_context= aug
        
        # Contrastive loss
        context = torch.cat((pre, aug), dim=0)
        b = context.size()[0]
        d = context.size()[1]
        context = context.view(b, d, 4*4*4)
        contra_context = self.Contrastiveblock(context)
        b, f, e = contra_context.size()
        contra_context = contra_context.view(b, f*e)
        pre, aug = contra_context.split(bs, 0)
        contra_loss = self.Contraloss(pre, aug)
        
        voxel = self.cnn(down_stream_context).squeeze()
        if len(voxel.size()) == 3:
            context = torch.unsqueeze(voxel, 0)
        # iou score
        iou_score = compute_iou_score(voxel, gt_volume)
        first_loss = F.mse_loss(torch.ones(iou_score.shape).type_as(iou_score), iou_score)
        # six-views score
        second_loss = compute_depth_score(voxel, gt_volume).squeeze()
        # context score
        third_loss = DiceLoss()(voxel, gt_volume)

        self.log('contra_loss/train', contra_loss.item())
        self.log('iou_score/train', iou_score.mean().item())
        self.log('first_loss/train', first_loss.item())
        self.log('second_loss/train', second_loss.item())
        self.log('third_loss/train', third_loss.item())

        # Put Replay Memory
        for img, aug, gt, tax_name, sam, score in zip(image, aug_image, gt_volume, tax, sample, iou_score): 
            self.memory.push(img.unsqueeze(0), aug.unsqueeze(0), gt.unsqueeze(0), tax_name, sam)
            self.tax_dict.add(tax_name)
        # Update Replay Memory
        tax_prob = []
        for tax_name in tax:
            prob = self.tax_dict.get_prob(tax_name)
            tax_prob.append(prob)
        tax_prob = torch.Tensor(tax_prob)
        idxs, _ = self.memory.sample(bs)
        td_errors = []
        for i, t in zip(iou_score, tax_prob):
            te = (1.0 - i.item()) + (1.0 - t.item())
            td_errors.append(te)
        self.memory.update(idxs, td_errors)

        loss = contra_loss + first_loss + second_loss + third_loss
        return loss

    def validation_step(self, batch, batch_idx):
        image, aug_image, gt_volume, tax, sample = batch
        # image.shape => [B, V, 3, H (224), W (224)]
        # gt_volume.shape => [B, 32, 32, 32]
        context = self.forward(aug_image)
        context = self.cnn(context).squeeze()
        context = self._polarize(torch.sigmoid(context))

        ious = []
        for gen, gt in zip(context, gt_volume):
            ious.append(compute_iou(gen, gt).item())

        self.log('val_iou', sum(ious) / len(ious))
        
        return ious, tax

    def validation_epoch_end(self, step_outputs):
        iou_dict = self.get_class_iou(step_outputs)

        for tax, iou in iou_dict.items():
            self.log(f'val_iou_{tax}', iou)

    def test_step(self, batch, batch_idx):
        image, aug_image, gt_volume, tax, sample = batch
        # image.shape => [B, V, 3, H (224), W (224)]
        # gt_volume.shape => [B, 32, 32, 32]
        context = self.forward(aug_image)
        context = self.cnn(context).squeeze()
        context = self._polarize(torch.sigmoid(context))
        
        # IOU
        ious = []
        for gen, gt in zip(context, gt_volume):
            ious.append(compute_iou(gen, gt).item())

        self.log('test_iou/test', sum(ious) / len(ious))

        return ious, tax

    def test_epoch_end(self, step_outputs):
        iou_dict = self.get_class_iou(step_outputs)

        for tax, iou in iou_dict.items():
            self.log(f'test_iou_{tax}', iou)

    def configure_optimizers(self):
        warmup_steps = self.cfg.optimization.warmup_steps
        def lambda_optim(epoch):
            lr_coeff = 1
            if self.global_step < warmup_steps:
                # training is in the warm-up phase, adjust learning rate linearly
                lr_coeff = (self.global_step + 1) / warmup_steps
            # make sure that LR doesn't go beyond 1
            return min(1, lr_coeff)

        base_lr = self.cfg.optimization.lr
        # Setup optimizer
        optimizer = torch.optim.Adagrad(filter(lambda p: p.requires_grad, self.parameters()), lr=base_lr)
        # Setup scheduler
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_optim)
        return (
            [optimizer],
            [
                {
                    # Warm-up scheduler should adjust the LR after each optimization step
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1
                }
            ]
        )

    def get_class_iou(self, step_outputs):
        # flatten step outputs
        tax_list = [tax for output in step_outputs for tax in output[1]]
        iou_list = [iou for output in step_outputs for iou in output[0]]

        iou_dict = {}

        for name in set(tax_list):
            # filter out ious for each class
            ious = [iou for iou, tax in zip(iou_list, tax_list) if tax == name]
            iou_dict[name] = sum(ious) / len(ious)

        iou_dict['mean'] = sum(list(iou_dict.values())) / len(iou_dict)

        return iou_dict

    def get_class_f1(self, fs, tax):
        # flatten step outputs
        tax_list = tax
        fs_list = fs

        f1_dict = {}

        for name in set(tax_list):
            # filter out ious for each class
            f1s = [f1 for f1, tax in zip(fs_list, tax_list) if tax == name]
            f1_dict[name] = sum(f1s) / len(f1s)

        f1_dict['mean'] = sum(list(f1_dict.values())) / len(f1_dict)

        return f1_dict