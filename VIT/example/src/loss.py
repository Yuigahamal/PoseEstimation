import torch
import torch.nn as nn
import torch.functional as F

class DETRLoss(nn.Module):
    """
    DETR的损失函数
    包含：匈牙利匹配 + 分类损失 + L1损失 + GIoU损失
    """

    def __init__(self,
                 num_classes=10,
                 weight_class = 1.0,
                 weight_bbox = 5.0,
                 weight_giou = 2.0):
        super(DETRLoss, self).__init__()
        self.num_classes = num_classes
        self.weight_class = weight_class
        self.weight_bbox = weight_bbox
        self.weight_giou = weight_giou

    def forward(self, pred_class, pred_bbox, gt_class, gt_bbox):
        """
        Args:
            pred_class: (B, 100, num_classes+1) 预测的类别logits
            pred_bbox:  (B, 100, 4) 预测的边界框 (cx, cy, w, h) 归一化
            gt_class:   list of (M_i,) 每个batch的真实类别
            gt_bbox:    list of (M_i, 4) 每个batch的真实边界框
        Returns:
            total_loss: 标量
            loss_dict: 包含各项损失的字典
        """

        B, N, _ = pred_class.shape

        # 对pred_class应用softmax得到概率
        pred_class = F.softmax(pred_class, dim=-1) # pred_class.shape = (B, N, num_classes+1)
        total_loss =  0.0
        loss_dict = {
            'loss_class': 0,
            'loss_bbox': 0,
            'loss_giou': 0
        }

        for b in range(B):
            pass


    def hungarian_matching(self,pred_prob, pred_bbox, gt_class, gt_bbox):
        """
        匈牙利匹配
        Args:
            pred_prob: (N, num_classes+1) 预测概率
            pred_bbox: (N, 4) 预测边界框(cx,cy,w,h)
            gt_class:  (M,) 真实类别
            gt_bbox:   (M, 4) 真实边界框
        Returns:
            pred_indices: (K,) 匹配上的预测索引
            gt_indices:   (K,) 匹配上的真实索引
        """
        N = pred_prob.shape[0]
        M = gt_class.shape[0]

        # 如果没有真实框，直接返回空匹配
        if M == 0:
            return torch.tensor([], device=pred_prob.device), torch.tensor([], device=gt_class.device)

        # 1. 分类代价：取预测中真实类别的概率的负对数
        class_cost = -pred_prob[:, gt_class]  # (N, M)

        # 2. 边界框L1代价
        l1_cost = torch.cdist(pred_bbox, gt_bbox, p=1)  # (N, M)
        # l1_cost[i][j]代表pred_bbox[i]与gt_bbox[j]之间的损失

        # 3. GIoU代价（负GIoU，因为要最小化代价）
        giou_cost = -self.giou_matrix(pred_bbox, gt_bbox)  # (N, M)

        # 4. 总代价（使用与损失相同的权重）




