import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from mmpose.registry import MODELS

@MODELS.register_module()
class KeyPointLoss(nn.Module):
    def __init__(self,
                 use_target_weight: bool = False,
                 loss_weight: float = 1.):
        super().__init__()
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def forward(self,
                output:Tensor,
                target:Tensor,
                target_weights:Tensor=None,)->Tensor:
        """
        Args:
            output (Tensor): 输出热力图[B, K, H, W]
            target (Tensor): 目标热力图 [B, K, H, W]
            target_weights (Tensor): 目标关键点级热力图权重 [B, K]，或者目标像素级热力图权重 [B, K, H, W]

        Returns:
            Tensor: 损失
        """
        if not self.use_target_weight:
            return F.mse_loss(output, target) * self.loss_weight

        assert target_weights is not None, "关键点权重为空"

        loss_per_pixel = F.mse_loss(output, target, reduction='none') # loss_per_pixel.shape = [B, K, H, W]

        if target_weights.dim() == 4:
            return (loss_per_pixel * target_weights).mean() * self.loss_weight

        loss_per_keypoint = loss_per_pixel.mean(dim=[2,3]) # loss_per_keypoint.shape = [B, K]
        loss = (loss_per_keypoint * target_weights).mean() * self.loss_weight
        return loss

