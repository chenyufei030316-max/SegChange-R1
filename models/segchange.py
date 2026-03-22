# -*- coding: utf-8 -*-
"""
@Project : SegChange-R1
@FileName: segchange.py
@Author  : ZhouFei
@Desc    : 适配北极RTS(冻土后退型热融滑塌)的遥感变化检测模型
@Note    : 1. 保留原始SegChange-R1核心能力
           2. 新增RTS物理约束：单时相掩码生成、物理一致性损失支持
           3. 支持Phi-1.5/bert文本编码器，BEV空间对齐，线性注意力
"""
from torch import nn
import torch
from models import (
    MaskHead, FeatureDiffModule, FPNFeatureFuser, LightweightFPN,
    DualInputVisualEncoder, TotalLoss, build_embs
)


class ChangeModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = self.cfg.device
        # RTS物理约束开关，配置文件可控制，默认开启
        self.use_rts_constraint = getattr(cfg.model, 'use_rts_constraint', True)

        # 1. 双输入视觉编码器：提取双时相多尺度BEV特征并空间对齐（SegChange-R1核心）
        self.dual_encoder = DualInputVisualEncoder(cfg).to(self.device)
        
        # 2. 特征差异模块：基于SE/CBAM计算双时相BEV特征差异，突出变化区域
        self.feature_diff = FeatureDiffModule(
            in_channels_list=self.cfg.model.out_dims,
            diff_attention=self.cfg.model.diff_attention
        ).to(self.device)
        
        # 3. FPN特征融合：支持标准FPN/轻量级L-FPN，可选EGA边缘增强/TokenConnector降维
        if self.cfg.model.fpn_type == 'FPN':
            self.fpn = FPNFeatureFuser(
                in_channels=self.cfg.model.out_dims,
                use_token_connector=self.cfg.model.use_token_connector,
                use_ega=self.cfg.model.use_ega
            ).to(self.device)
        elif self.cfg.model.fpn_type == 'L-FPN':
            self.fpn = LightweightFPN(
                in_channels=self.cfg.model.out_dims,
                use_token_connector=self.cfg.model.use_token_connector,
                use_ega=self.cfg.model.use_ega
            ).to(self.device)
        else:
            raise NotImplementedError(f"Unsupported FPN type: {self.cfg.model.fpn_type}")

        # 4. 文本编码器维度适配：Phi-1.5(2048维)/bert-base-uncased(768维)
        self.lang_dim = 2048 if cfg.model.text_encoder_name == 'microsoft/phi-1_5' else 768
        
        # 5. 掩码头：视觉-文本跨模态融合，生成像素级变化掩码（支持单时相BEV特征输入）
        self.mask_head = MaskHead(
            vis_dim=self.cfg.model.out_dims[-1] if not self.cfg.model.use_token_connector 
                    else self.cfg.model.out_dims[-1] // 2,
            lang_dim=self.lang_dim,
            num_classes=self.cfg.model.num_classes,
            n_heads=8
        ).to(self.device)

    def forward(self, image1: torch.Tensor, image2: torch.Tensor, embs: torch.Tensor):
        """
        前向传播：
        - 推理阶段：仅返回变化掩码mask_change
        - 训练阶段（开启RTS约束）：返回mask_change, mask_t1, mask_t2（用于RTS物理一致性损失）
        """
        # 设备统一，确保输入与模型在同一设备（cuda/cpu）
        image1 = image1.to(self.device)
        image2 = image2.to(self.device)
        
        # 步骤1：提取双时相多尺度BEV特征（已空间对齐，解决遥感影像配准误差）
        bev_feats1, bev_feats2 = self.dual_encoder(image1, image2)
        
        # 步骤2：提取多尺度特征差异 → FPN融合，整合小尺度细节+大尺度全局特征
        multi_scale_diff_feats = self.feature_diff(bev_feats1, bev_feats2)
        merged_feat = self.fpn(multi_scale_diff_feats)

        # 步骤3：生成核心变化掩码（RTS变化区域检测）
        mask_change = self.mask_head(merged_feat, embs)

        # 步骤4：RTS物理约束专属 - 训练阶段生成T1/T2单时相掩码
        # 基于1/32尺度BEV特征（bev_feats[-1]），空间一致性最优，用于计算面积/嵌套损失
        if self.training and self.use_rts_constraint:
            mask_t1 = self.mask_head(bev_feats1[-1], embs)  # T1时相冻土区掩码
            mask_t2 = self.mask_head(bev_feats2[-1], embs)  # T2时相滑塌区掩码
            return mask_change, mask_t1, mask_t2
        
        # 推理阶段仅返回变化掩码，与原始SegChange-R1完全兼容
        return mask_change

    def to(self, device):
        """重写设备迁移方法，确保所有子模块统一迁移到指定设备"""
        self.device = device
        self.dual_encoder = self.dual_encoder.to(device)
        self.feature_diff = self.feature_diff.to(device)
        self.fpn = self.fpn.to(device)
        self.mask_head = self.mask_head.to(device)
        return self


def build_model(cfg, training=False):
    """
    构建模型+损失函数
    :param cfg: 配置文件（yaml）
    :param training: True=训练模式（返回模型+损失），False=推理模式（仅返回模型）
    :return: 模型/（模型+损失）
    """
    model = ChangeModel(cfg)
    if not training:
        return model

    # 构建损失函数：支持RTS物理一致性损失（weight_rts_cons），直接传递完整损失配置
    losses = TotalLoss(
        num_classes=cfg.model.num_classes,
        config=cfg.loss  # 传递cfg.loss字典，兼容基础损失+RTS专属损失
    )

    return model, losses


# 测试代码：直接运行segchange.py即可验证模型完整性
if __name__ == '__main__':
    import torch
    import sys
    import os
    # 添加项目根目录到Python路径，解决utils导入问题
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
    from utils import load_config

    # 加载配置文件
    cfg = load_config("/hdd10Ta/chenyf/SegChange-R1/configs/config.yaml")
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    cfg.device = device  # 适配无cuda环境


    # 实例化模型
    model = ChangeModel(cfg).to(device)
    print(f"Model initialization complete. Running on device: {device}")

    # 生成测试数据：模拟双时相遥感影像+RTS文本嵌入
    batch_size = 2
    image1 = torch.randn(batch_size, 3, 256, 256).to(device)
    image2 = torch.randn(batch_size, 3, 256, 256).to(device)
    # RTS专属提示词
    rts_prompt = "Analyze the multi-temporal Arctic remote sensing images, infer the retrogressive thaw slump evolution types, detect the precise slump change boundaries and extract the spatial deformation features."
    embs = build_embs(
        prompts=[rts_prompt] * batch_size,
        text_encoder_name=cfg.model.text_encoder_name,
        freeze_text_encoder=cfg.model.freeze_text_encoder,
        device=device,
        batch_size=batch_size
    )

    # 推理模式测试
    model.eval()
    with torch.no_grad():
        mask = model(image1, image2, embs)
    print(f"Inference mode test passed. Output mask shape: {mask.shape} (Expected: [{batch_size}, 1, 512, 512])")

    # 训练模式测试
    model.train()
    with torch.no_grad():
        mask_change, mask_t1, mask_t2 = model(image1, image2, embs)
    print(f"Training mode (RTS constraints) test passed. mask_change: {mask_change.shape}, mask_t1: {mask_t1.shape}, mask_t2: {mask_t2.shape}")
    print("\nModel has no syntax errors, forward pass workflow is normal!")