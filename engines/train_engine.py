# -*- coding: utf-8 -*-
import pandas as pd
import os
import pprint
import numpy as np
import torch
import datetime
import logging
import random
import time
import warnings
from tensorboardX import SummaryWriter
from dataloader import build_dataset
from engines import train, evaluate
from models import build_model, PostProcessor
from utils import *

warnings.filterwarnings('ignore')
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class TrainingEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.device
        self.output_dir = get_output_dir(cfg.output_dir, cfg.name)
        self.logger = setup_logging(cfg, self.output_dir)
        self._setup_seed()
        self._setup_model()
        self._setup_optimizer()
        self._setup_dataloader()
        self._setup_postprocessor()
        self._setup_training_state()
        self._setup_logging_tools()
        self._resume_if_needed()

    def _setup_seed(self):
        seed = self.cfg.seed + get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def _setup_model(self):
        self.logger.info('------------------------ model params ------------------------')
        self.model, self.criterion = build_model(self.cfg, training=True)
        self.model.to(self.device)
        self.criterion.to(self.device)
        self.model_without_ddp = self.model
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info('number of params: %d', n_parameters)

    def _setup_optimizer(self):
        param_dicts = [
            {"params": [p for n, p in self.model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad], "lr": self.cfg.training.lr},
            {"params": [p for n, p in self.model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad], "lr": self.cfg.training.lr_backbone}
        ]
        self.optimizer = torch.optim.AdamW(param_dicts, lr=self.cfg.training.lr, weight_decay=self.cfg.training.weight_decay)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.cfg.training.lr_drop, gamma=0.1)

    def _setup_dataloader(self):
        self.dataloader_train, self.dataloader_val = build_dataset(cfg=self.cfg)

    def _setup_postprocessor(self):
        self.postprocessor = PostProcessor(min_area=2500, max_p_a_ratio=10, min_convexity=0.8)

    def _setup_training_state(self):
        self.start_epoch = self.cfg.training.start_epoch
        self.step = 0
        self.f1_list, self.iou_list, self.accuracy_list = [], [], []

    def _setup_logging_tools(self):
        tensorboard_dir = os.path.join(str(self.output_dir), 'tensorboard')
        os.makedirs(tensorboard_dir, exist_ok=True)
        self.writer = SummaryWriter(tensorboard_dir)
        self.csv_file_path = os.path.join(str(self.output_dir), 'result.csv')
        self.results_df = pd.DataFrame(columns=[
            'epoch', 'train_loss', 'train_oa', 
            'val_loss', 'val_f1', 'val_iou', 'val_oa'
        ])
        self.ckpt_dir = os.path.join(str(self.output_dir), 'checkpoints')
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def _resume_if_needed(self):
        if self.cfg.resume:
            self.logger.info(f'---------------- Resume from {self.cfg.resume} ----------------')
            checkpoint = torch.load(self.cfg.resume, map_location='cpu')
            self.model_without_ddp.load_state_dict(checkpoint['model'])
            if 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                self.start_epoch = checkpoint['epoch'] + 1

    def _train_one_epoch(self, epoch):
        t1 = time.time()
        # 物理约束 Warm-up 逻辑
        if epoch < 15:
            self.criterion.w_cons = 0.0
        else:
            self.criterion.w_cons = getattr(self.cfg.loss, 'weight_rts_cons', 0.5)

        stat = train(self.cfg, self.model, self.criterion, self.dataloader_train, self.optimizer, self.device, epoch)
        t2 = time.time()

        self.logger.info("[ep %d][lr %.7f] loss: %.4f, oa: %.4f, w_cons: %.2f, time: %.2fs",
                         epoch, self.optimizer.param_groups[0]['lr'], stat['loss'], stat['oa'], self.criterion.w_cons, t2-t1)
        
        if self.writer:
            self.writer.add_scalar('loss/total_loss', stat['loss'], epoch)
            self.writer.add_scalar('metric/train_oa', stat['oa'], epoch)
        
        return stat

    def _save_checkpoint(self, epoch, stat, name):
        checkpoint_path = os.path.join(self.ckpt_dir, f'{name}.pth')
        save_dict = {
            'model': self.model_without_ddp.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'epoch': epoch,
            'stat': stat,
            'cfg': self.cfg
        }
        torch.save(save_dict, checkpoint_path)

    def run(self):
        self.logger.info("------------------------ Start RTS Training ------------------------")
        start_time = time.time()
        for epoch in range(self.start_epoch, self.cfg.training.epochs):
            # 1. 训练
            train_stat = self._train_one_epoch(epoch)
            self.lr_scheduler.step()
            
            # 2. 保存每轮最新的模型
            self._save_checkpoint(epoch, train_stat, 'latest')

            # 3. 评估
            if epoch % self.cfg.training.eval_freq == 0:
                metrics = evaluate(self.cfg, self.model, self.criterion, self.postprocessor, self.dataloader_val, self.device, epoch)
                
                # 打印评估指标（这是你之前最想看到的部分）
                self.logger.info(
                    f"[ep {epoch}] [Validation] loss: {metrics['loss']:.4f}, f1: {metrics['f1']:.4f}, "
                    f"iou: {metrics['iou']:.4f}, oa: {metrics['oa']:.4f} ---- @Best IoU: {np.max(self.iou_list) if self.iou_list else 0:.4f}"
                )
                
                # 写入 Tensorboard
                if self.writer:
                    self.writer.add_scalar('metric/val_iou', metrics['iou'], epoch)
                    self.writer.add_scalar('metric/val_f1', metrics['f1'], epoch)

                # 更新列表并保存最佳模型
                self.f1_list.append(metrics['f1'])
                self.iou_list.append(metrics['iou'])
                if metrics['f1'] == np.max(self.f1_list):
                    self._save_checkpoint(epoch, train_stat, 'best_f1')
                if metrics['iou'] == np.max(self.iou_list):
                    self._save_checkpoint(epoch, train_stat, 'best_iou')

                # 4. 保存到 CSV
                new_row = {
                    'epoch': epoch, 'train_loss': train_stat['loss'], 'train_oa': train_stat['oa'],
                    'val_loss': metrics['loss'], 'val_f1': metrics['f1'], 'val_iou': metrics['iou'], 'val_oa': metrics['oa']
                }
                self.results_df = pd.concat([self.results_df, pd.DataFrame([new_row])], ignore_index=True)
                self.results_df.to_csv(self.csv_file_path, index=False)

        total_time = time.time() - start_time
        self.logger.info(f'Training finished. Total time: {datetime.timedelta(seconds=int(total_time))}')
        self.logger.info(f"Best F1: {np.max(self.f1_list):.4f}, Best IoU: {np.max(self.iou_list):.4f}")