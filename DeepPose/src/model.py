"""
DeepPose: 基于深度神经网络的人体姿态估计
DeepPose: Deep Neural Network for Human Pose Estimation

参考论文:
    "DeepPose: Human Pose Estimation via Deep Neural Networks"
    Alexander Toshev & Christian Szegedy, CVPR 2014
    (https://arxiv.org/abs/1312.4659)

架构概览:
    Stage 1 (初始阶段):  全图 → CNN主干网络 → 全连接层 → 2K 个关节坐标
    Stages 2+ (级联阶段): 关节裁剪区域 → 共享CNN → FC → 2K 个坐标偏移 → 精细化
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
#  AlexNet 风格的主干网络（DeepPose 的特征提取器）
# ---------------------------------------------------------------------------

class AlexNetFeatures(nn.Module):
    """AlexNet 的卷积层部分，针对姿态估计任务进行了适配。

    在末尾使用 AdaptiveAvgPool2d，使得无论输入图像尺寸如何，
    输出的特征图始终为 (256, 6, 6)。原始 DeepPose 期望 220×220 的输入，
    但自适应池化使模型更加灵活。

    输出: (B, 256, 6, 6) → 展平后得到 9216 维的特征向量。
    """

    def __init__(self, output_spatial: int = 6):
        """初始化 AlexNet 特征提取器。

        Args:
            output_spatial: 自适应池化后的空间尺寸（默认 6，即 6×6 的特征图）
        """
        super().__init__()
        self.features = nn.Sequential(
            # conv1: 步长 4，无填充（AlexNet 风格）
            # 输入 (3, H, W) → 输出 (96, H/4, W/4)
            nn.Conv2d(3, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            # 局部响应归一化（LRN），增强相邻特征间的竞争
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            # 最大值池化，进一步下采样
            nn.MaxPool2d(kernel_size=3, stride=2),

            # conv2: 5×5 卷积，padding=2 保持空间尺寸
            # 输入 96 通道 → 输出 256 通道
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # conv3: 3×3 卷积，padding=1 保持空间尺寸
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # conv4: 3×3 卷积，与 conv3 通道数相同
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # conv5: 3×3 卷积，降回 256 通道，后接最大值池化
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # 自适应平均池化：将任意大小的特征图统一缩放到 (output_spatial, output_spatial)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((output_spatial, output_spatial))
        self.out_channels = 256          # 输出通道数
        self.output_spatial = output_spatial  # 输出空间尺寸

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入图像张量，形状为 (B, 3, H, W)

        Returns:
            特征图张量，形状为 (B, 256, output_spatial, output_spatial)
        """
        x = self.features(x)          # 通过卷积层序列
        x = self.adaptive_pool(x)     # 自适应池化到固定尺寸
        return x


# ---------------------------------------------------------------------------
#  可微分的关节区域裁剪
# ---------------------------------------------------------------------------

def extract_joint_crops(
    x: torch.Tensor,
    joints: torch.Tensor,
    crop_ratio: float = 0.25,
    crop_size: int = 64,
) -> torch.Tensor:
    """根据预测的关节坐标，通过 grid_sample 从原图中提取关节周围的图像块。

    此操作完全可微分，因此梯度可以通过裁剪操作反向传播到
    上一阶段的坐标预测。

    Args:
        x:          输入图像，形状为 (B, C, H, W)
        joints:     关节坐标，范围为 [0, 1]，形状为 (B, K, 2)，最后一维为 (x, y)
        crop_ratio: 裁剪区域相对于原图尺寸的边长比例
        crop_size:  每个裁剪块的输出空间尺寸（正方形）

    Returns:
        crops: 提取的关节图像块，形状为 (B*K, C, crop_size, crop_size)
    """
    B, C, H, W = x.shape
    K = joints.shape[1]        # 关键点数量
    device = x.device

    # ---- 构建基础采样网格：(crop_h, crop_w, 2)，值域为 [-1, 1] --------
    # meshgrid 生成 crop_size × crop_size 的规则网格
    gy, gx = torch.meshgrid(
        torch.linspace(-1, 1, crop_size, device=device),   # y 方向（行）
        torch.linspace(-1, 1, crop_size, device=device),   # x 方向（列）
        indexing='ij',  # 矩阵索引：第一维是行(y)，第二维是列(x)
    )
    base_grid = torch.stack([gx, gy], dim=-1)                   # (S, S, 2)
    base_grid = base_grid * crop_ratio                           # 缩放到裁剪范围
    base_grid = base_grid.unsqueeze(0).unsqueeze(0)              # (1, 1, S, S, 2)

    # ---- 将每个裁剪区域平移到其对应的关节位置上 ---------------------------
    joints_norm = joints * 2.0 - 1.0                             # [0,1] → [-1,1]
    grid = base_grid + joints_norm.unsqueeze(2).unsqueeze(3)     # (B, K, S, S, 2)
    grid = grid.view(B * K, crop_size, crop_size, 2)

    # ---- 扩展图像以支持每个关节独立采样 ------------------------------
    # 将输入图像沿关节维度复制 K 份
    x_expanded = (
        x.unsqueeze(1)                     # (B, 1, C, H, W)
        .expand(-1, K, -1, -1, -1)        # (B, K, C, H, W)
        .reshape(B * K, C, H, W)          # (B*K, C, H, W)
    )

    # 使用双线性插值从扩展图像中采样每个关节周围的区域
    crops = F.grid_sample(x_expanded, grid, mode='bilinear',
                          align_corners=True)
    return crops  # (B*K, C, S, S)


# ---------------------------------------------------------------------------
#  Stage 1 — 初始（粗略）姿态预测
# ---------------------------------------------------------------------------

class InitialStage(nn.Module):
    """全图 → 主干卷积特征 → 全连接回归 → 2K 个关节坐标。

    这是 DeepPose 的第一阶段，从完整图像中预测所有关节位置的
    初始粗略估计。
    """

    def __init__(
        self,
        backbone: nn.Module,
        feat_dim: int,
        num_keypoints: int,
        hidden_dim: int = 4096,
        dropout: float = 0.5,
    ):
        """初始化第一阶段。

        Args:
            backbone:       卷积特征提取器（如 AlexNetFeatures）
            feat_dim:       展平后的特征维度（AlexNet → 256×6×6 = 9216）
            num_keypoints:  关键点数量（如 LSP 数据集为 14）
            hidden_dim:     全连接隐藏层维度（默认 4096）
            dropout:        Dropout 概率，用于防止过拟合
        """
        super().__init__()
        self.backbone = backbone
        self.num_keypoints = num_keypoints

        # 全连接回归器：特征 → 隐藏层 → 隐藏层 → 2K 个坐标
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),         # 第1层: 特征维度 → 4096
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),                     # 随机丢弃 50% 神经元
            nn.Linear(hidden_dim, hidden_dim),       # 第2层: 4096 → 4096
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2 * num_keypoints), # 输出层: 4096 → 2K
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入图像，形状为 (B, 3, H, W)

        Returns:
            关节坐标，形状为 (B, K, 2)，值域为 [0, 1]
        """
        feats = self.backbone(x)       # 提取卷积特征
        feats = feats.flatten(1)       # 展平为 (B, feat_dim)
        out = self.fc(feats)           # 全连接回归
        out = torch.sigmoid(out)       # 使用 sigmoid 将输出约束到 [0, 1]
        return out.view(-1, self.num_keypoints, 2)  # 重塑为 (B, K, 2)


# ---------------------------------------------------------------------------
#  级联精炼阶段（Stage 2, 3, …）
# ---------------------------------------------------------------------------

class CascadeStage(nn.Module):
    """级联精炼：关节区域裁剪 → 共享CNN → 偏移量预测。

    对于每个关节，围绕上一阶段的预测位置裁剪一个图像块。
    所有图像块由共享的轻量级 CNN 独立处理，然后将各关节特征
    拼接起来，通过全连接层联合预测所有关节的精炼偏移量。
    """

    def __init__(
        self,
        num_keypoints: int,
        crop_ratio: float = 0.25,
        crop_size: int = 64,
    ):
        """初始化级联阶段。

        Args:
            num_keypoints: 关键点数量
            crop_ratio:    裁剪区域相对于原图的比例
            crop_size:     裁剪块的输出像素尺寸
        """
        super().__init__()
        self.num_keypoints = num_keypoints
        self.crop_ratio = crop_ratio
        self.crop_size = crop_size

        # ---- 用于处理关节图像块的轻量级共享 CNN --------------------
        # 三个卷积块，每个通过 MaxPool2d(2) 将空间尺寸减半
        self.conv = nn.Sequential(
            # Block 1: 3→32 通道, 空间尺寸 /2（stride=2）, 池化再 /2
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                     # → 空间尺寸 /4

            # Block 2: 32→64 通道, 尺寸不变, 池化 /2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                     # → 空间尺寸 /8

            # Block 3: 64→128 通道, 尺寸不变, 池化 /2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                     # → 空间尺寸 /16
        )

        # 经过 3 次 2× 下采样后的空间尺寸
        self.feat_spatial = crop_size // 16   # 例: 64 // 16 = 4
        self.feat_dim_per_joint = 128 * self.feat_spatial * self.feat_spatial  # 每个关节的特征维度

        # 全连接回归器：拼接所有关节特征 → 预测 2K 个偏移量
        self.fc = nn.Sequential(
            nn.Linear(self.feat_dim_per_joint * num_keypoints, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 2 * num_keypoints),  # 输出每个关节的 (dx, dy)
        )

    def forward(
        self, x: torch.Tensor, joints: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """前向传播。

        Args:
            x:      输入图像，形状为 (B, C, H, W)
            joints: 上一阶段预测的关节坐标，形状为 (B, K, 2)，值域 [0, 1]

        Returns:
            deltas:  预测的偏移量，形状为 (B, K, 2)
            refined: 精炼后的关节坐标，形状为 (B, K, 2)，裁剪到 [0, 1]
        """
        B = x.size(0)
        K = self.num_keypoints

        # 提取每个关节周围的图像块并独立处理
        crops = extract_joint_crops(x, joints, self.crop_ratio, self.crop_size)
        feats = self.conv(crops)                                 # (B*K, 128, S/16, S/16)
        feats = feats.flatten(1)                                 # (B*K, feat_dim_per_joint)

        # 拼接所有关节的特征，联合预测偏移量
        feats = feats.view(B, K * self.feat_dim_per_joint)       # (B, K * feat_dim)
        deltas = self.fc(feats)                                  # (B, 2K)
        # 使用 tanh 将偏移量限制在 [-crop_ratio, crop_ratio] 范围内，
        # 防止单步偏移过大导致偏离目标
        deltas = torch.tanh(deltas) * self.crop_ratio
        deltas = deltas.view(B, K, 2)

        # 坐标更新: 当前坐标 + 预测偏移量
        refined = joints + deltas
        refined = torch.clamp(refined, 0.0, 1.0)                 # 裁剪到有效范围

        return deltas, refined


# ---------------------------------------------------------------------------
#  DeepPose — 完整级联模型
# ---------------------------------------------------------------------------

class DeepPose(nn.Module):
    """DeepPose: 用于人体姿态估计的级联回归网络。

    参数
    ----------
    num_keypoints : int
        人体关节数量（如 LSP=14、FLIC=16、COCO=17）
    backbone : nn.Module, optional
        Stage 1 的 CNN 特征提取器，默认为 AlexNetFeatures
    feat_dim : int
        主干网络展平后的特征维度。
        AlexNetFeatures → 256×6×6 = 9216
    num_cascade_stages : int
        精炼阶段的数量（总阶段数 = 1 + num_cascade_stages）
    crop_ratio : float
        级联阶段的相对裁剪尺寸（占图像边长的比例）
    crop_size : int
        每个关节裁剪块的绝对输出尺寸（像素）
    hidden_dim : int
        Stage 1 全连接隐藏层的维度
    dropout : float
        Stage 1 全连接层的 Dropout 概率
    """

    def __init__(
        self,
        num_keypoints: int = 14,               # 默认 LSP 数据集的 14 个关节
        backbone: Optional[nn.Module] = None,
        feat_dim: int = 9216,                  # 256 × 6 × 6
        num_cascade_stages: int = 2,            # 默认 2 个级联阶段
        crop_ratio: float = 0.25,
        crop_size: int = 64,
        hidden_dim: int = 4096,
        dropout: float = 0.5,
    ):
        """初始化 DeepPose 模型。"""
        super().__init__()
        self.num_keypoints = num_keypoints
        self.num_cascade_stages = num_cascade_stages

        # 如果未指定主干网络，默认使用 AlexNet 特征提取器
        if backbone is None:
            backbone = AlexNetFeatures(output_spatial=6)

        # ---- Stage 1: 初始粗略预测 -----------------------------------
        self.stage1 = InitialStage(
            backbone=backbone,
            feat_dim=feat_dim,
            num_keypoints=num_keypoints,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # ---- 级联精炼阶段 -----------------------------------------------
        # ModuleList 确保每个阶段都有独立的参数
        self.cascade_stages = nn.ModuleList([
            CascadeStage(num_keypoints, crop_ratio, crop_size)
            for _ in range(num_cascade_stages)
        ])

        self._init_weights()  # 参数初始化

    def _init_weights(self):
        """Kaiming 初始化卷积层权重；正态分布初始化全连接层权重。

        卷积层使用 Kaiming（He）初始化以适配 ReLU 激活函数，
        全连接层使用均值 0、标准差 0.01 的正态分布初始化。
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """前向传播，返回所有阶段的预测结果。

        Args:
            x: 输入图像，形状为 (B, 3, H, W)

        Returns:
            outputs: 列表，包含每个阶段的 (B, K, 2) 预测张量，
                     列表长度 = 1 + num_cascade_stages
        """
        outputs: list[torch.Tensor] = []

        # ---- Stage 1: 初始粗略预测 ----------------------------------
        joints = self.stage1(x)
        outputs.append(joints)

        # ---- 级联阶段: 逐阶段精炼 ----------------------------------
        for cascade in self.cascade_stages:
            _, joints = cascade(x, joints)   # 取 refined 坐标，丢弃 delta
            outputs.append(joints)

        return outputs

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """仅返回最终（最精炼）的预测结果。

        Args:
            x: 输入图像，形状为 (B, 3, H, W)

        Returns:
            joints: 最终关节坐标，形状为 (B, K, 2)，值域 [0, 1]
        """
        outputs = self.forward(x)
        return outputs[-1]  # 取最后一阶段的输出


# ---------------------------------------------------------------------------
#  损失函数
# ---------------------------------------------------------------------------

def deeppose_loss(
    predictions: list[torch.Tensor],
    targets: torch.Tensor,
    stage_weights: Optional[list[float]] = None,
) -> torch.Tensor:
    """所有级联阶段的加权 MSE（均方误差）损失之和。

    DeepPose 的核心思想之一：对每个阶段的预测都施加监督，
    而非仅监督最终输出。这确保了中间阶段的预测也是合理的。

    Args:
        predictions:   DeepPose.forward() 返回的预测列表，每个元素为 (B, K, 2)
        targets:       真实关节坐标（标签），形状为 (B, K, 2)，值域 [0, 1]
        stage_weights: 每个阶段的损失权重。None 表示均匀权重 [1, 1, …]

    Returns:
        标量损失值
    """
    if stage_weights is None:
        stage_weights = [1.0] * len(predictions)

    total = 0.0
    for pred, w in zip(predictions, stage_weights):
        total = total + w * F.mse_loss(pred, targets)
    return torch.Tensor(total)


# ---------------------------------------------------------------------------
#  便捷构造函数
# ---------------------------------------------------------------------------

def deeppose_alexnet(
    num_keypoints: int = 14,
    num_cascade_stages: int = 2,
    pretrained: Optional[str] = None,
) -> DeepPose:
    """使用 AlexNet 主干网络的 DeepPose（忠实于原论文）。

    参数
    ----------
    num_keypoints : int
        关节数量（LSP=14, FLIC=16）
    num_cascade_stages : int
        级联精炼阶段的数量
    pretrained : Optional[str]
        预训练模型权重的路径（可选）

    Returns:
        配置好的 DeepPose 模型实例
    """
    backbone = AlexNetFeatures(output_spatial=6)
    model = DeepPose(
        num_keypoints=num_keypoints,
        backbone=backbone,
        feat_dim=256 * 6 * 6,          # 9216
        num_cascade_stages=num_cascade_stages,
    )
    if pretrained is not None:
        _load_pretrained(model, pretrained)  # 加载预训练权重
    return model


def deeppose_resnet50(
    num_keypoints: int = 14,
    num_cascade_stages: int = 2,
    pretrained_backbone: bool = True,
    pretrained: Optional[str] = None,
) -> DeepPose:
    """使用 ResNet-50 主干网络的 DeepPose（现代化变体）。

    用 ImageNet 预训练的 ResNet-50 替代原始的 AlexNet，
    以获得更强的特征提取能力。ResNet 在分类器之前被截断，
    并配备自适应池化，使特征维度始终为 2048。

    参数
    ----------
    num_keypoints : int
        关节数量
    num_cascade_stages : int
        级联精炼阶段的数量
    pretrained_backbone : bool
        是否加载 ImageNet 预训练权重到主干网络
    pretrained : Optional[str]
        完整模型的预训练权重路径（可选）

    Returns:
        配置好的 DeepPose 模型实例
    """
    import torchvision.models as tv_models

    # 加载 ResNet-50 主干网络（可选择是否使用 ImageNet 预训练权重）
    resnet = tv_models.resnet50(
        weights='DEFAULT' if pretrained_backbone else None,
    )
    # 保留除最后分类器外的所有层
    backbone = nn.Sequential(
        *list(resnet.children())[:-1],           # → (B, 2048, 1, 1)
        nn.Flatten(1),                           # → (B, 2048)
    )

    # 我们需要让主干网络返回一个可以被 InitialStage.flatten(1) 处理的张量。
    # 这里将其包装为返回 (B, 2048, 1, 1) 的形式，使 InitialStage 的 flatten 成为无操作。
    class _ResNetBackbone(nn.Module):
        def __init__(self, trunk):
            super().__init__()
            self.trunk = trunk  # 包含 avgpool + flatten 的 ResNet 主干

        def forward(self, x):
            # trunk 内部已通过 ResNet 的 avgpool + Flatten 完成了池化和展平
            # → (B, 2048)。将其展开为 (B, 2048, 1, 1)，
            # 使 InitialStage.flatten(1) 成为无害操作（2048*1*1 = 2048）。
            return self.trunk(x).unsqueeze(-1).unsqueeze(-1)

    model = DeepPose(
        num_keypoints=num_keypoints,
        backbone=_ResNetBackbone(backbone),
        feat_dim=2048,                            # ResNet-50 的特征维度
        num_cascade_stages=num_cascade_stages,
    )
    if pretrained is not None:
        _load_pretrained(model, pretrained)
    return model


def _load_pretrained(model: DeepPose, path: str) -> None:
    """从磁盘加载预训练模型权重。

    兼容多种保存格式：
    - 直接保存的 state_dict
    - 包含 'state_dict' 键的训练检查点
    - 包含 'model' 键的训练检查点

    Args:
        model: DeepPose 模型实例
        path:  预训练权重文件的路径
    """
    state = torch.load(path, map_location='cpu', weights_only=True)
    # 兼容不同的检查点保存格式
    if 'state_dict' in state:
        state = state['state_dict']
    elif 'model' in state:
        state = state['model']
    # strict=False: 允许部分加载（如仅加载主干网络权重）
    model.load_state_dict(state, strict=False)


# ---------------------------------------------------------------------------
#  快速测试（直接运行 python model.py 即可验证模型是否可用）
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # 测试 AlexNet 版本
    print("=" * 60)
    print("DeepPose — AlexNet 主干网络")
    print("=" * 60)
    model = deeppose_alexnet(num_keypoints=14, num_cascade_stages=2)
    model.eval()

    # 创建模拟输入: 2 张 3 通道 220×220 的图像（原论文的输入尺寸）
    dummy = torch.randn(2, 3, 220, 220)
    with torch.no_grad():
        outputs = model(dummy)

    print(f"输入形状: {dummy.shape}")
    for i, out in enumerate(outputs):
        print(f"Stage {i+1} 输出形状: {out.shape}  (B, K, 2)")
    print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    # 损失函数正常性检查
    targets = torch.rand(2, 14, 2)
    loss = deeppose_loss(outputs, targets)
    print(f"损失 (MSE): {loss.item():.6f}")

    # 测试 predict() 方法（仅返回最终结果）
    final = model.predict(dummy)
    print(f"predict() 输出形状: {final.shape}")

    print()
    print("=" * 60)
    print("DeepPose — ResNet-50 主干网络")
    print("=" * 60)
    try:
        # pretrained_backbone=False: 不使用 ImageNet 预训练，仅测试结构
        model_rn = deeppose_resnet50(num_keypoints=14, num_cascade_stages=1,
                                     pretrained_backbone=False)
        model_rn.eval()
        with torch.no_grad():
            outputs_rn = model_rn(dummy)
        print(f"输入形状: {dummy.shape}")
        for i, out in enumerate(outputs_rn):
            print(f"Stage {i+1} 输出形状: {out.shape}")
        print(f"参数量: {sum(p.numel() for p in model_rn.parameters()) / 1e6:.2f} M")
    except ImportError:
        print("torchvision 不可用 — 跳过 ResNet 测试。")
