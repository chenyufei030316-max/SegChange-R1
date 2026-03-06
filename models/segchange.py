# -*- coding: utf-8 -*-
from torch import nn
import torch
# 确保从你的模型库中导入更新后的 TotalLoss
from models import MaskHead, FeatureDiffModule, FPNFeatureFuser, LightweightFPN, DualInputVisualEncoder, TotalLoss, \
    build_embs

class ChangeModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = self.cfg.device
        self.dual_encoder = DualInputVisualEncoder(cfg).to(self.device)
        # SegChange-R1 核心：通过 BEV 转换模块将双时相特征对齐 [cite: 9, 31, 103]
        self.feature_diff = FeatureDiffModule(in_channels_list=self.cfg.model.out_dims, 
                                              diff_attention=self.cfg.model.diff_attention).to(self.device)
        
        if self.cfg.model.fpn_type == 'FPN':
            self.fpn = FPNFeatureFuser(in_channels=self.cfg.model.out_dims, 
                                       use_token_connector=self.cfg.model.use_token_connector,
                                       use_ega=self.cfg.model.use_ega).to(self.device)
        elif self.cfg.model.fpn_type == 'L-FPN':
            self.fpn = LightweightFPN(in_channels=self.cfg.model.out_dims, 
                                      use_token_connector=self.cfg.model.use_token_connector,
                                      use_ega=self.cfg.model.use_ega).to(self.device)
        else:
            raise NotImplementedError(f"Unsupported FPN type: {self.cfg.model.fpn_type}")

        self.lang_dim = 2048 if cfg.model.text_encoder_name == 'microsoft/phi-1_5' else 768
        self.mask_head = MaskHead(
            vis_dim=self.cfg.model.out_dims[-1] if not self.cfg.model.use_token_connector else self.cfg.model.out_dims[-1]//2,
            lang_dim=self.lang_dim,
            num_classes=self.cfg.model.num_classes,
            n_heads=8
        ).to(self.device)

    def forward(self, image1: torch.Tensor, image2: torch.Tensor, embs: torch.Tensor):
        image1 = image1.to(self.device)
        image2 = image2.to(self.device)
        
        # 1. 编码器提取双时相 BEV 特征 [cite: 83, 105]
        # multi_scale_bev_feats 已经是空间对齐后的特征 [cite: 151]
        bev_feats1, bev_feats2 = self.dual_encoder(image1, image2)
        
        # 2. 特征差异提取与融合 [cite: 144, 145]
        multi_scale_diff_feats = self.feature_diff(bev_feats1, bev_feats2)
        merged_feat = self.fpn(multi_scale_diff_feats)

        # 3. 生成最终变化图 [cite: 146]
        mask_change = self.mask_head(merged_feat, embs)

        # 4. [针对 RTS 物理约束新增] 分别生成 T1 和 T2 的预测掩码
        # 利用最后一层 BEV 特征进行单时相预测，用于计算面积和嵌套损失
        if self.training:
            # 这里的 bev_feats1[-1] 对应 1/32 尺度的对齐特征 [cite: 90]
            mask_t1 = self.mask_head(bev_feats1[-1], embs)
            mask_t2 = self.mask_head(bev_feats2[-1], embs)
            return mask_change, mask_t1, mask_t2
        
        return mask_change

    def to(self, device):
        self.device = device
        self.dual_encoder = self.dual_encoder.to(device)
        self.feature_diff = self.feature_diff.to(device)
        self.fpn = self.fpn.to(device)
        self.mask_head = self.mask_head.to(device)
        return self

def build_model(cfg, training=False):
    model = ChangeModel(cfg)
    if not training:
        return model

    # 5. 更新损失函数：引入 RTS 物理一致性权重 [cite: 8]
    # 这里通过字典形式传递配置，确保 TotalLoss 能读取到 weight_rts_cons
    losses = TotalLoss(
        num_classes=cfg.model.num_classes,
        config=cfg.loss  # 直接传递配置字典
    )

    return model, losses