from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmpose.registry import MODELS
from mmpose.models.backbones.base_backbone import  BaseBackbone
from mmpose.models import BaseHead
from mmpose.utils.typing import OptSampleList, OptConfigType, Features, Predictions
from torch import Tensor


@MODELS.register_module()
class VIT(BaseBackbone):
    def __init__(self, img_size=(224,224),
                 patch_size=16,
                 embed_dim = 768,
                 depth=12,
                 num_heads=12,
                 qkv_bias=False,
                 qkv_scale=True,
                 attn_weight_dropout_ratio=0.1,
                 attn_proj_dropout_ratio=0.1,
                 fnn_ratio=4,
                 fnn_drop_ratio=0.1,
                 encoder_dropout_ratio=0.1,
                 pos_dropout_ratio=0.1,):
        super(VIT, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.seq_len = img_size[0] // patch_size * img_size[1] // patch_size
        self.embed_dim = embed_dim

        self.patch_embedding = PatchEmbedding(patch_size, embed_dim)

        self.pos_embedding = nn.Parameter(torch.zeros(1, self.seq_len, self.embed_dim))
        self.pos_drop = nn.Dropout(pos_dropout_ratio)

        self.encoders = nn.Sequential(*[
            Encoder(embed_dim, num_heads, qkv_bias, qkv_scale, attn_weight_dropout_ratio, attn_proj_dropout_ratio, fnn_ratio, fnn_drop_ratio, encoder_dropout_ratio)
            for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self,m):
        """
        ViT 权重初始化函数
        通过 model.apply(_init_vit_weights) 遍历所有子模块
        """
        if isinstance(m, nn.Linear):
            # 线性层：截断正态分布初始化，std=0.02
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)  # 偏置初始化为0

        elif isinstance(m, nn.Conv2d):
            # 卷积层：截断正态分布初始化
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            # LayerNorm：权重初始化为1，偏置初始化为0
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self,x):

        x = self.patch_embedding(x) # x.shape =[B,seq_len,dim]
        x = self.pos_drop(x + self.pos_embedding) # x.shape =[B,seq_len,dim]

        x = self.encoders(x) # x.shape = [B,seq_len,dim]
        x = self.norm(x) # x.shape =[B,seq_len,dim]


        # 当head用MMPOSE中自带的HeatmapHead时，HeatmapHead需要(B,dim,H,W)所以要进行维度处理，当用自己的Decoder时，去掉下面的操作
        x = torch.transpose(x,dim0=1,dim1=2) # x.shape = [B,embed_dim,seq_len]
        x = torch.reshape(x,(x.shape[0],x.shape[1],self.img_size[0] // self.patch_size,self.img_size[1] // self.patch_size))
        return tuple([x]) # HeatmapHead接受的是一个多尺度的热图元组，且取最后一个即分辨率最高的热图作为输入，所以这里要返回元组

@MODELS.register_module()
class Decoder(BaseHead):
    def __init__(self, embed_dim=768, num_keypoints=17,upscale_factor=4,decoder_type="simple"):
        super(Decoder, self).__init__()
        self.decoder =SimpleDecoder(embed_dim, num_keypoints=num_keypoints,upscale_factor=upscale_factor) \
            if decoder_type=='simple' \
            else ClassicDecoder(embed_dim, num_keypoints=num_keypoints)

        self.apply(self._init_weights)
    def _init_weights(self,m):
        """
        Decoder 权重初始化函数
        通过 model.apply(_init_vit_weights) 遍历所有子模块
        """
        if isinstance(m, nn.Linear):
            # 线性层：截断正态分布初始化，std=0.02
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)  # 偏置初始化为0

        elif isinstance(m, nn.Conv2d):
            # 卷积层：截断正态分布初始化
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            # LayerNorm：权重初始化为1，偏置初始化为0
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)


    def forward(self, feats: Tuple[Tensor]):
        """
        前向传播

        Args:
            feats: backbone输出的多尺度特征图，形状为 ([B, C, H, W]),类型为Tuple

        Returns:
            Tensor: 预测的热力图，形状为 [B, num_keypoints, H_out, W_out]
        """
        feat = feats[-1]
        return self.decoder(feat)

    def loss(self,
             feats: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: OptConfigType = {}) -> dict:
        """
        计算损失函数

        Args:
            feats: backbone输出的特征图，形状为 [B, C, H, W]
            batch_data_samples: 包含真实标签的数据样本列表，列表是PoseDataSample类型,该类型包含以下数据:
            {
                gt_instances：
                {
                   keypoints: 形状为 (N, K, 2) 的张量，代表 N 个实例的 J 个关键点的 (x, y) 坐标,
                   keypoint_weights: 形状为 (N, K) 的张量，表示每个关键点的权重。这个信息在计算损失时非常重要，可以用来忽略那些被遮挡或未标注的关键点，这个操作被称为 keypoint_weights
                }，
                gt_fields (真值场)：
                {
                    heatmaps: 形状为 (K, H, W) 的热力图张量，其中 J 是关键点数量。
                }，
                metainfo(数据元信息)：
                {
                    img_shape:图像尺寸
                    crop_size：裁剪尺寸
                    heatmap_size:热力图尺寸
                    Decode:解码器
                }

            }

            train_cfg: 训练配置

        Returns:
            dict: 损失字典
        """
        # 1. 前向传播得到预测热力图
        pred_fields = self.forward(feats)  # [B, num_keypoints, H_out, W_out]

        # 2. 获取真实热力图和权重
        # torch.cat 像是在现有的维度上“拼接”，张量的总维度数不变。
        # torch.stack 像是在新的维度上“堆叠”，张量的总维度数会增加 1。
        gt_heatmaps = torch.stack(
            [d.gt_fields.heatmaps for d in batch_data_samples])
        keypoint_weights = torch.cat([
            d.gt_instance_labels.keypoint_weights for d in batch_data_samples
        ])

        # 3. 计算损失
        losses = dict()
        loss = self.loss_module(pred_fields, gt_heatmaps, keypoint_weights)

        losses.update(loss_kpt=loss)

        return losses

        heatmap = self.forward(feats)
        losses = dict()



    def predict(self,
                feats: Features,
                batch_data_samples: OptSampleList,
                test_cfg: OptConfigType = {}) -> Predictions:
        pass





class ClassicDecoder(nn.Module):
    """
    经典解码器（Classic Decoder）：
    由两个转置卷积（Deconvolution/ConvTranspose）模块堆叠而成，每个模块包含一个上采样层、批归一化（BN）和ReLU激活函数。
    特征图会被上采样4倍，最后通过一个1x1卷积层输出热图。
    """

    def __init__(self, embed_dim=768, num_keypoints=17,
                 deconv_channels=[256, 256], deconv_kernel=[4, 4]):
        """
        Args:
            embed_dim: 编码器输出维度 (如768)
            num_keypoints: 关键点数量 (COCO:17, MPII:16)
            deconv_channels: 每个转置卷积的输出通道数
            deconv_kernel: 每个转置卷积的卷积核大小
        """
        super(ClassicDecoder, self).__init__()

        # 将序列特征重塑为2D特征图前的投影层
        self.projection = nn.Conv2d(embed_dim, deconv_channels[0], kernel_size=1)

        # 构建多个转置卷积层进行上采样
        deconv_layers = []
        in_channels = deconv_channels[0]
        stride = 2

        for i, (out_channels, kernel_size) in enumerate(zip(deconv_channels, deconv_kernel)):
            output_padding = 1 if kernel_size % 2  else 0
            padding = (kernel_size - stride + output_padding) // 2

            deconv_layers.append(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,  # 上采样2倍
                    padding=padding,
                    output_padding=output_padding,
                    bias=False
                )
            )
            deconv_layers.append(nn.BatchNorm2d(out_channels))
            deconv_layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        self.deconv_layers = nn.Sequential(*deconv_layers)

        # 最终预测层：输出关键点热图
        self.final_layer = nn.Conv2d(
            in_channels=deconv_channels[-1],
            out_channels=num_keypoints,
            kernel_size=1
        )

    def forward(self, x, img_size=(256, 192),patch_size=16):
        """
        Args:
            x: 编码器输出 [B, seq_len, embed_dim]
            img_size: 原始输入图像尺寸 (height, width)
        Returns:
            heatmaps: 关键点热图 [B, num_keypoints, H, W]
        """
        B, seq_len, embed_dim = x.shape

        # 计算特征图尺寸 (假设patch_size=16)
        h = img_size[0] // patch_size  # 例如: 256//16=16
        w = img_size[1] // patch_size  # 例如: 192//16=12

        # 1. 重塑为2D特征图
        x = x.permute(0, 2, 1)  # [B, embed_dim, seq_len]
        x = x.reshape(B, embed_dim, h, w)  # [B, embed_dim, h, w]

        # 2. 投影到解码器通道数
        x = self.projection(x)  # [B, deconv_channels[0], h, w]

        # 3. 转置卷积上采样
        x = self.deconv_layers(x)  # [B, deconv_channels[-1], h*4, w*4]

        # 4. 预测热图
        heatmaps = self.final_layer(x)  # [B, num_keypoints, H, W]

        return heatmaps

class SimpleDecoder(nn.Module):
    """
    简单解码器：使用双线性插值上采样
    速度更快，性能略低于经典解码器
    """

    def __init__(self, embed_dim=768, num_keypoints=17,
                 upscale_factor=4, hidden_dim=256):
        """
        Args:
            embed_dim: 编码器输出维度
            num_keypoints: 关键点数量
            upscale_factor: 上采样倍数 (通常为4)
            hidden_dim: 中间隐藏层维度
        """
        super(SimpleDecoder, self).__init__()

        self.upscale_factor = upscale_factor

        # 序列到2D的投影
        self.projection = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        # 最终预测层
        self.final_layer = nn.Conv2d(
            hidden_dim, num_keypoints, kernel_size=1
        )

    def forward(self, x, img_size=(192, 256),patch_size = 16):
        B, seq_len, embed_dim = x.shape

        # 计算特征图尺寸
        h = img_size[0] // patch_size
        w = img_size[1] // patch_size

        # 重塑为2D特征图
        x = x.permute(0, 2, 1)
        x = x.reshape(B, embed_dim, h, w)

        # 投影
        x = self.projection(x)  # [B, hidden_dim, h, w]

        # 双线性插值上采样
        x = F.interpolate(
            x,
            scale_factor=self.upscale_factor, # 上采样倍数
            mode='bilinear', # 线性插值
            align_corners=False  # 像素按面积对齐，更符合图像信号处理理论，在 CNN 解码器中更常用
        )  # [B, hidden_dim, h*4, w*4]

        # 预测热图
        heatmaps = self.final_layer(x)  # [B, num_keypoints, H, W]

        return heatmaps

class Encoder(nn.Module):
    """
    一个Encoder
    由两个LayerNorm，一个自注意力层，两个个dropout，，一个MLP，和残差连接组成
    """
    def __init__(self,
                 dim, # 输入词向量的维度
                 num_heads = 12, # 自注意力层头数
                 qkv_bias = False, # 自注意力层生成qkv时是否使用偏置
                 qkv_scale = True ,# 自注意力层计算qkv分数时是否使用缩放
                 attn_weight_dropout_ratio = 0.1, # 自注意力层权重归一化后的dropout率
                 attn_proj_dropout_ratio = 0.1, # 自注意力层最后输出线性层的dropout率,
                 fnn_ratio=4,
                 fnn_drop_ratio=0.1,
                 encoder_dropout_ratio=0.1,
                ):
        super(Encoder, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention =Attention(dim, num_heads, qkv_bias, qkv_scale, attn_weight_dropout_ratio, attn_proj_dropout_ratio)
        self.dropout = nn.Dropout(encoder_dropout_ratio) if encoder_dropout_ratio > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.fnn = FNN(in_features=dim,hide_features=dim*fnn_ratio,out_features=dim,dropout_ratio=fnn_drop_ratio)


    def forward(self,x):
        x_ = x
        x = self.norm1(x) # x.shape = [B,seq_len,dim]
        x = self.attention(x) # x.shape = [B,seq_len,dim]
        x = self.dropout(x) # x.shape = [B,seq_len,dim]
        x = x+x_


        x_ = x
        x = self.norm2(x) # x.shape = [B,seq_len,dim]
        x = self.fnn(x) # x.shape = [B,seq_len,dim]
        x = self.dropout(x) # x.shape = [B,seq_len,dim]
        x = x+x_


        return x

class PatchEmbedding(nn.Module):
    """
    将图像 patches 化为 token
    例如CIFAR-10 输入图像尺寸为 32x32
    通过PathEmbedding层 输出 序列长度(seq_len)为4,维度(word_vector)为256的序列数据
    """

    def __init__(self,patch_size = 16,embed_dim = 768):
        super(PatchEmbedding,self).__init__()
        # 运用卷积将图像变成序列数据,输出通道就是序列数据的向量维度，即词向量的维度
        self.proj = nn.Conv2d(3,embed_dim,kernel_size=patch_size,stride=patch_size)

        # 层归一化
        self.layer_norm = nn.LayerNorm(embed_dim)


    def forward(self,x):
        x = self.proj(x) # x.shape  = [B,embed_dim,H,W]
        B,embed_dim,H,W = x.shape
        x = torch.flatten(x,start_dim=2) # x.shape = [B,embed_dim,seq_len]
        x = torch.permute(x, (0, 2, 1)) # x.shape = [B,seq_len,embed_dim] 调整维度顺序
        # print(x.shape)
        x = self.layer_norm(x) # x.shape = [B,seq_len,embed_dim]

        return x

class Attention(nn.Module):
    """
    多头自注意力层
    用于构建Encoder
    """
    def __init__(self,
                 dim, # 输入词向量的维度
                 num_heads = 12, # 头数
                 qkv_bias = False, # 生成qkv时是否使用偏置
                 qkv_scale = True ,# 计算qkv分数时是否使用缩放
                 weight_dropout_ratio = 0.1, # 权重归一化后的dropout率
                 proj_dropout_ratio = 0.1 # 最后输出线性层的dropout率
                 ):
        super(Attention, self).__init__()
        self.num_heads = num_heads # 头数
        self.scale = (dim//num_heads) ** -0.5 if qkv_scale else 1.0 # 缩放参数,即Transformer论文中 sqrt(d_k)
        self.q = nn.Linear(dim, dim, bias=qkv_bias) # 用于生成q的线性层
        self.k = nn.Linear(dim, dim, bias=qkv_bias) # 用于生成k的线性层
        self.v = nn.Linear(dim, dim, bias=qkv_bias) # 用于生成v的线性层
        self.proj = nn.Linear(dim,dim)
        self.proj_drop = nn.Dropout(proj_dropout_ratio) if proj_dropout_ratio > 0 else nn.Identity()
        self.weight_drop = nn.Dropout(weight_dropout_ratio) if weight_dropout_ratio > 0 else nn.Identity()

    def forward(self,x):
        B,seq_len,dim = x.shape

        q = self.q(x) # q.shape = [B,seq_len,dim]
        k = self.k(x) # k.shape = [B,seq_len,dim]
        v = self.v(x) # v.shape = [B,seq_len,dim]

        q = torch.reshape(q, (B, seq_len, self.num_heads, dim // self.num_heads))
        k = torch.reshape(k, (B, seq_len, self.num_heads, dim // self.num_heads))
        v = torch.reshape(v, (B, seq_len, self.num_heads, dim // self.num_heads))

        q = torch.permute(q, (0, 2, 1, 3)) # q.shape = [B,num_heads,seq_len,dim//num_heads]
        k = torch.permute(k, (0, 2, 1, 3)) # k.shape = [B,num_heads,seq_len,dim//num_heads]
        v = torch.permute(v, (0, 2, 1, 3)) # v.shape = [B,num_heads,seq_len,dim//num_heads]

        score = q @ torch.transpose(k, -2, -1)  * self.scale # score.shape = [B,num_heads,seq_len,seq_len]

        weight = torch.softmax(score, dim=-1) # weight.shape = [B,num_heads,seq_len,seq_len],权重归一化
        weight = self.weight_drop(weight)

        attention = weight @ v # attention.shape = [B,num_heads,seq_len,dim//num_heads]
        attention = torch.transpose(attention, 1, 2) # attention.shape = [B,seq_len,num_heads,dim//num_heads]
        attention = torch.reshape(attention, (B, seq_len, dim)) # attention.shape = [B,seq_len,dim]
        attention = self.proj(attention) # attention.shape = [B,seq_len,dim]
        attention = self.proj_drop(attention) # attention.shape = [B,seq_len,dim]
        return attention

class FNN(nn.Module):
    """
    前馈神经网络层
    用于构建Encoder
    """
    def __init__(self,in_features,hide_features=None,out_features=None,dropout_ratio=0.1):
        super().__init__()
        self.hide_features = hide_features or in_features
        self.out_features =  out_features or in_features
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(in_features,self.hide_features)
        self.fc2 = nn.Linear(self.hide_features,self.out_features)
        self.dropout = nn.Dropout(dropout_ratio)


    def forward(self,x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    vit = VIT(img_size=(192,256))
    head = Decoder()
    x= torch.randn(1, 3, 192, 256)
    x = vit(x)
    print(type(x))



