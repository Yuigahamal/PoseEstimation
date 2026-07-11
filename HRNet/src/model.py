"""
HRNet (High-Resolution Network) for Human Pose Estimation

Reference:
    "Deep High-Resolution Representation Learning for Visual Recognition"
    (https://arxiv.org/abs/1908.07919)

This implementation supports HRNet-W32, HRNet-W48, and custom width configurations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
#  Conv helpers
# ---------------------------------------------------------------------------

def conv3x3(in_channels, out_channels, stride=1):
    """3x3 convolution with padding 1."""
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride,
        padding=1, bias=False,
    )


def conv1x1(in_channels, out_channels, stride=1):
    """1x1 convolution."""
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=1, stride=stride, bias=False,
    )


# ---------------------------------------------------------------------------
#  Residual blocks
# ---------------------------------------------------------------------------

class BasicBlock(nn.Module):
    """Basic residual block: 3x3-conv → BN → ReLU → 3x3-conv → BN  [+ shortcut]."""

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    """Bottleneck residual block: 1x1 → 3x3 → 1x1  [+ shortcut]."""

    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        width = out_channels
        self.conv1 = conv1x1(in_channels, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = conv3x3(width, width, stride)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(width, out_channels * self.expansion)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


# ---------------------------------------------------------------------------
#  HRNet building blocks
# ---------------------------------------------------------------------------

class HighResolutionModule(nn.Module):
    """
    One multi-resolution module of HRNet.

    It runs several residual blocks on parallel branches (each operating at a
    different resolution), then fuses information across branches via
    up/down-sampling.
    """

    def __init__(
        self,
        num_branches: int,
        block: type,
        num_blocks: list[int],
        num_channels: list[int],
        fuse_method: str = 'SUM',
        multi_scale_output: bool = True,
    ):
        super().__init__()
        self.num_branches = num_branches
        self.fuse_method = fuse_method

        # --- Per-branch residual layers ----------------------------------
        self.branches = nn.ModuleList()
        for i in range(num_branches):
            self.branches.append(
                self._make_branch(block, num_blocks[i], num_channels[i])
            )

        # --- Multi-scale fusion layers (one per output branch) -----------
        self.fuse_layers = nn.ModuleList()
        for i in range(num_branches if multi_scale_output else 1):
            fuse_layer = nn.ModuleList()
            for j in range(num_branches):
                if j > i:          # j is lower-res than i  →  upsample
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels[j], num_channels[i],
                                kernel_size=1, stride=1, bias=False,
                            ),
                            nn.BatchNorm2d(num_channels[i]),
                            nn.Upsample(scale_factor=2 ** (j - i), mode='nearest'),
                        )
                    )
                elif j == i:       # same resolution  →  identity (None)
                    fuse_layer.append(nn.Identity())
                else:              # j is higher-res than i  →  downsample
                    ops = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            ops.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_channels[j], num_channels[i],
                                        kernel_size=3, stride=2, padding=1, bias=False,
                                    ),
                                    nn.BatchNorm2d(num_channels[i]),
                                )
                            )
                        else:
                            ops.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_channels[j], num_channels[j],
                                        kernel_size=3, stride=2, padding=1, bias=False,
                                    ),
                                    nn.BatchNorm2d(num_channels[j]),
                                    nn.ReLU(inplace=True),
                                )
                            )
                    fuse_layer.append(nn.Sequential(*ops))
            self.fuse_layers.append(fuse_layer)

        self.relu = nn.ReLU(inplace=True)

    @staticmethod
    def _make_branch(block, num_blocks, num_channels):
        layers = []
        for _ in range(num_blocks):
            layers.append(block(num_channels, num_channels))
        return nn.Sequential(*layers)

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        # Safety: prune branch count when fed fewer tensors than branches
        # (happens in Stage-2 when previous stage had only 1 branch)
        if self.num_branches > len(x):
            x = x[: self.num_branches]

        # --- Pass each branch through its residual blocks -----------------
        x = [branch(xi) for branch, xi in zip(self.branches, x)]

        # --- Multi-scale fusion ------------------------------------------
        x_fused = []
        for fuse_layer in self.fuse_layers:
            fused = None
            for j, layer in enumerate(fuse_layer):  # type: ignore
                y = layer(x[j])
                if fused is None:
                    fused = y
                else:
                    if self.fuse_method == 'SUM':
                        fused = fused + y
                    elif self.fuse_method == 'CAT':
                        fused = torch.cat([fused, y], dim=1)
            x_fused.append(self.relu(fused))
        return x_fused


# ---------------------------------------------------------------------------
#  Transition layers  (add a new lower-resolution branch)
# ---------------------------------------------------------------------------

def _make_transition_layer(
    num_branches_pre: int,
    num_channels_pre: list[int],
    num_channels_cur: list[int],
    block: type,
    num_blocks: list[int],
):
    """
    Build a transition that takes branches from previous stage and adds one
    new, lower-resolution branch. Returns (transition_layers, new_num_channels).
    """
    transition_layers = nn.ModuleList()

    # Branches staying unchanged
    for i in range(num_branches_pre):
        transition_layers.append(
            nn.Sequential(
                nn.Conv2d(num_channels_pre[i], num_channels_cur[i],
                          kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(num_channels_cur[i]),
                nn.ReLU(inplace=True),
            )
        )

    # New lower-resolution branch (created via stride-2 conv from the
    # lowest-resolution pre-branch)
    i_lowest_pre = num_branches_pre - 1
    transition_layers.append(
        nn.Sequential(
            nn.Conv2d(num_channels_pre[i_lowest_pre], num_channels_cur[-1],
                      kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_channels_cur[-1]),
            nn.ReLU(inplace=True),
        )
    )
    return transition_layers


# ---------------------------------------------------------------------------
#  Stage builders
# ---------------------------------------------------------------------------

def _make_stage(
    stage_cfg: dict,
    num_channels_pre: list[int],
):
    """
    Build one HRNet stage: transition (if adding branches) + N modules.

    Parameters
    ----------
    stage_cfg : dict with keys:
        num_modules, num_branches, block, num_blocks, num_channels, fuse_method
    num_channels_pre : list[int] – channel counts from previous stage
    """
    num_branches_pre = len(num_channels_pre)
    num_branches_cur = stage_cfg['NUM_BRANCHES']
    block = stage_cfg['BLOCK']
    num_blocks_cur = stage_cfg['NUM_BLOCKS']
    num_channels_cur = stage_cfg['NUM_CHANNELS']
    fuse_method = stage_cfg.get('FUSE_METHOD', 'SUM')

    modules = nn.ModuleList()

    # --- Transition (if branch count grows) ------------------------------
    if num_branches_cur > num_branches_pre:
        transition = _make_transition_layer(
            num_branches_pre, num_channels_pre,
            num_channels_cur, block, num_blocks_cur,
        )
        modules.append(transition)
        num_channels_pre = num_channels_cur  # after transition, channels match

    # --- High-resolution modules -----------------------------------------
    for i in range(stage_cfg['NUM_MODULES']):
        multi_scale_output = (i != stage_cfg['NUM_MODULES'] - 1)
        modules.append(
            HighResolutionModule(
                num_branches=num_branches_cur,
                block=block,
                num_blocks=num_blocks_cur,
                num_channels=num_channels_pre,
                fuse_method=fuse_method,
                multi_scale_output=multi_scale_output,
            )
        )

    return modules, num_channels_pre


# ---------------------------------------------------------------------------
#  HRNet  (full model for pose estimation)
# ---------------------------------------------------------------------------

class HRNet(nn.Module):
    """
    High-Resolution Network for human pose estimation.

    Parameters
    ----------
    num_keypoints : int
        Number of keypoints to predict (e.g. 17 for COCO).
    width : int, optional
        Base width of the network. Default 32 → HRNet-W32.
        Use 48 for HRNet-W48.
    """

    def __init__(self, num_keypoints: int = 17, width: int = 32):
        super().__init__()

        # --- Stem: two stride-2 convs → 1/4 resolution -------------------
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # --- Stage 1: single high-resolution branch (Bottleneck blocks) --
        # Like ResNet-50 layer1: 4 Bottleneck blocks, hidden dim = 64
        channels_stage1 = 256
        self.stage1 = nn.Sequential(
            *self._make_layer(Bottleneck, 64, 64, 4),
        )
        # After Bottleneck: 64 → 256
        self.transition1 = nn.Sequential(
            nn.Conv2d(channels_stage1, width, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )

        # --- Stage 2 -----------------------------------------------------
        self.stage2_cfg = {
            'NUM_MODULES': 1,
            'NUM_BRANCHES': 2,
            'NUM_BLOCKS': [4, 4],
            'NUM_CHANNELS': [width, width * 2],
            'BLOCK': BasicBlock,
            'FUSE_METHOD': 'SUM',
        }
        num_channels = [width]
        self.stage2, num_channels = _make_stage(self.stage2_cfg, num_channels)

        # --- Stage 3 -----------------------------------------------------
        self.stage3_cfg = {
            'NUM_MODULES': 4,
            'NUM_BRANCHES': 3,
            'NUM_BLOCKS': [4, 4, 4],
            'NUM_CHANNELS': [width, width * 2, width * 4],
            'BLOCK': BasicBlock,
            'FUSE_METHOD': 'SUM',
        }
        self.stage3, num_channels = _make_stage(self.stage3_cfg, num_channels)

        # --- Stage 4 -----------------------------------------------------
        self.stage4_cfg = {
            'NUM_MODULES': 3,
            'NUM_BRANCHES': 4,
            'NUM_BLOCKS': [4, 4, 4, 4],
            'NUM_CHANNELS': [width, width * 2, width * 4, width * 8],
            'BLOCK': BasicBlock,
            'FUSE_METHOD': 'SUM',
        }
        self.stage4, num_channels = _make_stage(self.stage4_cfg, num_channels)

        # --- Final layer: heatmap prediction -----------------------------
        # Upsample all branches to the highest-res scale, concat, then 1x1 conv
        # The highest-res branch is at index 0, with num_channels[0]
        final_in_channels = sum(num_channels)
        self.final_layer = nn.Conv2d(
            final_in_channels, num_keypoints, kernel_size=1, stride=1,
        )

        self._init_weights()

    @staticmethod
    def _make_layer(block, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                conv1x1(in_channels, out_channels * block.expansion, stride),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = [block(in_channels, out_channels, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(block(out_channels * block.expansion, out_channels))
        return layers

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, 3, H, W)  input image.

        Returns:
            heatmaps : (B, K, H/4, W/4)  predicted keypoint heatmaps.
        """
        # --- Stem ---------------------------------------------------------
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # --- Stage 1 ------------------------------------------------------
        x = self.stage1(x)
        x = self.transition1(x)
        x_list = [x]  # single branch

        # --- Stage 2 ------------------------------------------------------
        for m in self.stage2:
            if isinstance(m, nn.ModuleList):         # transition
                x_list = [layer(x_list[i if i < len(x_list) else -1])
                          for i, layer in enumerate(m)]
            else:                                     # HR module
                x_list = m(x_list)

        # --- Stage 3 ------------------------------------------------------
        for m in self.stage3:
            if isinstance(m, nn.ModuleList):
                x_list = [layer(x_list[i if i < len(x_list) else -1])
                          for i, layer in enumerate(m)]
            else:
                x_list = m(x_list)

        # --- Stage 4 ------------------------------------------------------
        for m in self.stage4:
            if isinstance(m, nn.ModuleList):
                x_list = [layer(x_list[i if i < len(x_list) else -1])
                          for i, layer in enumerate(m)]
            else:
                x_list = m(x_list)

        # --- Fuse & predict -----------------------------------------------
        # Upsample lower-resolution branches to the highest-res (branch 0)
        h, w = x_list[0].size(2), x_list[0].size(3)
        upsampled = []
        for i, feat in enumerate(x_list):
            if i == 0:
                upsampled.append(feat)
            else:
                upsampled.append(
                    F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=True)
                )
        fused = torch.cat(upsampled, dim=1)
        out = self.final_layer(fused)
        return out


# ---------------------------------------------------------------------------
#  Convenience constructors
# ---------------------------------------------------------------------------

def hrnet_w32(num_keypoints: int = 17, pretrained: str | None = None) -> HRNet:
    """HRNet-W32 model."""
    model = HRNet(num_keypoints=num_keypoints, width=32)
    if pretrained is not None:
        _load_pretrained(model, pretrained)
    return model


def hrnet_w48(num_keypoints: int = 17, pretrained: str | None = None) -> HRNet:
    """HRNet-W48 model."""
    model = HRNet(num_keypoints=num_keypoints, width=48)
    if pretrained is not None:
        _load_pretrained(model, pretrained)
    return model


def _load_pretrained(model: HRNet, path: str):
    state = torch.load(path, map_location='cpu', weights_only=True)
    # Support both raw state-dict and checkpoint-wrapper formats
    if 'state_dict' in state:
        state = state['state_dict']
    elif 'model' in state:
        state = state['model']
    model.load_state_dict(state, strict=False)


# ---------------------------------------------------------------------------
#  Quick test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    model = hrnet_w32(num_keypoints=17)
    model.eval()

    dummy = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        out = model(dummy)

    print(f"Input  shape: {dummy.shape}")
    print(f"Output shape: {out.shape}")   # expected: (1, 17, 64, 64)
    print(f"Parameters:  {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
