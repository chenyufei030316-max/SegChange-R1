import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. 基础分类损失 (BCE With Logits)
# ==========================================
class BCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, targets):
        # pred: [B, 1, H, W], targets: [B, H, W]
        loss = F.binary_cross_entropy_with_logits(pred.squeeze(1), targets.float())
        return loss

# ==========================================
# 2. 边界增强损失 (Dice Loss)
# ==========================================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, num_classes=1):
        super().__init__()
        self.smooth = smooth
        self.num_classes = num_classes

    def forward(self, pred, targets):
        if self.num_classes == 1:
            pred = torch.sigmoid(pred).squeeze(1)
            targets = targets.float()
            intersection = (pred * targets).sum()
            union = pred.sum() + targets.sum()
            dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        else:
            pred = F.softmax(pred, dim=1)
            targets = F.one_hot(targets.long(), num_classes=self.num_classes).permute(0, 3, 1, 2).float()
            intersection = (pred * targets).sum(dim=(2, 3))
            union = pred.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
            dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth).mean()

        return 1.0 - dice_score

# ==========================================
# 3. 类别均衡损失 (Focal Loss)
# ==========================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, num_classes=1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, pred, targets):
        if self.num_classes == 1:
            pred = pred.squeeze(1)
            pt = torch.sigmoid(pred)
            pt = pt * targets + (1 - pt) * (1 - targets)
            focal_loss = -self.alpha * (1.0 - pt) ** self.gamma * torch.log(pt + 1e-6)
        else:
            logpt = F.log_softmax(pred, dim=1)
            pt = torch.exp(logpt)
            targets = targets.unsqueeze(1)
            logpt = torch.gather(logpt, 1, targets).squeeze(1)
            pt = torch.gather(pt, 1, targets).squeeze(1)
            focal_loss = -self.alpha * (1.0 - pt) ** self.gamma * logpt
        return focal_loss.mean()

# ==========================================
# 4. 特征区分度损失 (Batch Balanced Contrastive Loss)
# ==========================================
class BatchBalancedContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0, num_classes=1):
        super().__init__()
        self.margin = margin
        self.num_classes = num_classes

    def forward(self, pred, targets):
        pred = pred.squeeze(1)
        targets = targets.float()
        distance = torch.abs(pred - targets)
        pos_mask = (targets == 1)
        neg_mask = (targets == 0)
        
        
        num_pos = pos_mask.sum()
        num_neg = neg_mask.sum()

        # 增加 1e-6 防止除以 0
        loss_pos = (distance * pos_mask).sum() / (num_pos + 1e-6)
        loss_neg = (torch.clamp(self.margin - distance, min=0.0) * neg_mask).sum() / (num_neg + 1e-6)

       
        return loss_pos + loss_neg

# ==========================================
# 5. [核心新增] RTS 物理一致性损失 (Consistency Loss)
# ==========================================
class RTSConsistencyLoss(nn.Module):
    """
    针对 RTS 不可逆扩张特性的约束。
    依赖 SegChange-R1 的 BEV 模块完成的空间对齐。
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred_t1, pred_t2):
        # 将 Logits 转为概率
        prob_t1 = torch.sigmoid(pred_t1)
        prob_t2 = torch.sigmoid(pred_t2)

        # 1. 面积单调性：T1 面积不应大于 T2 (RTS 只有扩张或不变)
        area_t1 = prob_t1.sum(dim=(1, 2, 3))
        area_t2 = prob_t2.sum(dim=(1, 2, 3))
        loss_area = torch.mean(F.relu(area_t1 - area_t2))

        # 2. 空间嵌套：T1 区域应被 T2 包含 (解决迁移时的位置漂移噪声)
        # 惩罚在 T1 存在但在 T2 消失的滑塌像素
        loss_inclusion = torch.mean(prob_t1 * (1.0 - prob_t2))

        return loss_area + loss_inclusion

# ==========================================
# 6. 完整组合损失函数 (Total Loss)
# ==========================================
class TotalLoss(nn.Module):
    def __init__(self, num_classes=1, config=None):
        super().__init__()
        # 将 .get('key', default) 改为使用 getattr(object, 'key', default)
        self.w_ce = getattr(config, 'weight_ce', 1.0)
        self.w_dice = getattr(config, 'weight_dice', 3.0)
        self.w_focal = getattr(config, 'weight_focal', 1.5)
        self.w_bcl = getattr(config, 'weight_bcl', 0.3)
        self.w_cons = getattr(config, 'weight_rts_cons', 0.5)

        self.cls_loss = BCEWithLogitsLoss() if num_classes == 1 else CrossEntropyLoss()
        self.dice_loss = DiceLoss(num_classes=num_classes)
        # 注意：这里如果 FocalLoss 内部也用了 .get，也需要同步修改
        self.focal_loss = FocalLoss(
            alpha=getattr(config, 'alpha', 0.5), 
            gamma=getattr(config, 'gamma', 2.0), 
            num_classes=num_classes
        )
        self.bcl_loss = BatchBalancedContrastiveLoss(num_classes=num_classes)
        self.cons_loss = RTSConsistencyLoss()

    def forward(self, pred_final, targets, pred_t1=None, pred_t2=None):
        """
        pred_final: SegChange-R1 最终的变化预测图
        targets: Ground Truth
        pred_t1, pred_t2: 模型对 T1, T2 原始时相生成的中间掩码预测 (需修改 forward 导出)
        """
        # 基础视觉任务损失 [cite: 8, 153]
        ce = self.cls_loss(pred_final, targets)
        dice = self.dice_loss(pred_final, targets)
        focal = self.focal_loss(pred_final, targets)
        bcl = self.bcl_loss(pred_final, targets)

        total_loss = (self.w_ce * ce + self.w_dice * dice + 
                      self.w_focal * focal + self.w_bcl * bcl)

        # 物理演化一致性损失 (利用 BEV 对齐后的空间一致性) [cite: 31, 238]
        if pred_t1 is not None and pred_t2 is not None:
            total_loss += self.w_cons * self.cons_loss(pred_t1, pred_t2)

        return total_loss