import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self,
                 dim, # 输入词向量的维度
                 num_heads = 12, # 自注意力头数
                 attn_qkv_bias = False,
                 attn_qkv_scale = True, # 自注意力qkv缩放
                 attn_weight_dropout_ratio = 0, # 自注意力权重dropout率
                 attn_proj_dropout_ratio = 0, # 自注意力输出线性层dropout率
                 feed_dropout_ratio = 0, # 前馈网络dropout率
                 feed_ratio = 4, # 前馈网络隐藏层维度比例
                 encoder_dropout_ratio=0, # 编码器dropout率
                 ):
        super(Encoder, self).__init__()
        self.attn = Attention(dim, num_heads, qkv_bias=attn_qkv_bias, qkv_scale=attn_qkv_scale, weight_dropout_ratio=attn_weight_dropout_ratio, proj_dropout_ratio=attn_proj_dropout_ratio)
        self.norm1 = nn.LayerNorm(dim)

        self.dropout = nn.Dropout(encoder_dropout_ratio) if encoder_dropout_ratio > 0 else nn.Identity()

        self.ffn = FeedForward(dim, dim * feed_ratio, dim, feed_dropout_ratio)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self,x):
        _x = x

        x = self.norm1(x)
        x = self.attn(x)
        x = self.dropout(x)
        x = x + _x
        _x = x


        x = self.norm2(x)
        x = self.ffn(x)
        x = x + _x

        return x

class Decoder(nn.Module):
    def __init__(self,
                 dim, # 输入词向量的维度
                 attn_num_heads = 8,
                 attn_qkv_bias=False,
                 attn_qkv_scale=True,
                 attn_weight_dropout_ratio = 0,
                 attn_proj_dropout_ratio = 0,
                 cross_num_heads = 8,
                 cross_attn_qkv_scale=True,
                 cross_weight_dropout_ratio = 0,
                 cross_proj_dropout_ratio = 0,
                 feed_dropout_ratio = 0, # dropout率
                 feed_ratio=4,
                 decoder_attn_dropout_ratio=0,
                 decoder_cross_attn_dropout_ratio=0,
                 ):
        super(Decoder, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, attn_num_heads, qkv_bias=attn_qkv_bias, qkv_scale=attn_qkv_scale, weight_dropout_ratio=attn_weight_dropout_ratio, proj_dropout_ratio=attn_proj_dropout_ratio)
        self.attn_dropout = nn.Dropout(decoder_attn_dropout_ratio) if decoder_attn_dropout_ratio > 0 else nn.Identity()

        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(dim, cross_num_heads, qkv_scale=cross_attn_qkv_scale, weight_dropout_ratio=cross_weight_dropout_ratio, proj_dropout_ratio=cross_proj_dropout_ratio)
        self.cross_attn_dropout = nn.Dropout(decoder_cross_attn_dropout_ratio) if decoder_cross_attn_dropout_ratio > 0 else nn.Identity()

        self.norm3 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, dim * feed_ratio, dim, feed_dropout_ratio)

    def forward(self,h,x):
        # 自注意力
        _x = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.attn_dropout(x)
        x = _x + x  # 残差连接

        # 交叉注意力
        _x = x
        x = self.norm2(x)
        x = self.cross_attn(x, h, h)
        x = self.cross_attn_dropout(x)
        x = _x + x  # 残差连接

        # FFN
        _x = x
        x = self.norm3(x)
        x = self.ffn(x)
        x = _x + x  # 残差连接

        return x

class Attention(nn.Module):
    """
    多头自注意力层
    用于构建Encoder和Decoder
    """
    def __init__(self,
                 dim, # 输入词向量的维度
                 num_heads = 12, # 头数
                 qkv_bias = False, # 生成qkv时是否使用偏置
                 qkv_scale = True ,# 计算qkv分数时是否使用缩放
                 weight_dropout_ratio = 0, # 权重归一化后的dropout率
                 proj_dropout_ratio = 0 # 最后输出线性层的dropout率
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

class CrossAttention(nn.Module):
    """
    交叉注意力层
    """
    def __init__(self,dim,
                 num_heads=8,  # 头数
                 qkv_scale=True,
                 weight_dropout_ratio=0,  # 权重归一化后的dropout率
                 proj_dropout_ratio=0,  # 最后输出线性层的dropout率
                ):
        super(CrossAttention, self).__init__()
        self.qkv_scale = qkv_scale
        self.scale = (dim // num_heads) ** -0.5 if qkv_scale else 1.0
        self.num_heads = num_heads
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_dropout_ratio) if proj_dropout_ratio > 0 else nn.Identity()
        self.weight_drop = nn.Dropout(weight_dropout_ratio) if weight_dropout_ratio > 0 else nn.Identity()

    def forward(self,q,k,v):
        """
        q: [B, D_seq_len, dim]   (来自解码器)
        k: [B, E_seq_len, dim]   (来自编码器图像特征)
        v: [B, E_seq_len, dim]   (来自编码器图像特征)
        """
        B,D_seq_len,dim = q.shape # [B,E_seq_len,dim]
        E_seq_len = k.shape[1]

        q = torch.reshape(q, (B, D_seq_len, self.num_heads, dim // self.num_heads))
        k = torch.reshape(k, (B, E_seq_len, self.num_heads, dim // self.num_heads))
        v = torch.reshape(v, (B, E_seq_len, self.num_heads, dim // self.num_heads))

        q = torch.permute(q, (0, 2, 1, 3)) # q.shape = [B,num_heads,D_seq_len,dim//num_heads]
        k = torch.permute(k, (0, 2, 1, 3)) # k.shape = [B,num_heads,E_seq_len,dim//num_heads]
        v = torch.permute(v, (0, 2, 1, 3)) # v.shape = [B,num_heads,E_seq_len,dim//num_heads]

        score = q @ torch.transpose(k, -2, -1) * self.scale  # score.shape = [B,num_heads,D_seq_len,E_seq_len]

        weight = torch.softmax(score, dim=-1)  # weight.shape = [B,num_heads,D_seq_len,E_seq_len],权重归一化
        weight = self.weight_drop(weight)

        attention = weight @ v  # attention.shape = [B,num_heads,D_seq_len,dim//num_heads]
        attention = torch.transpose(attention, 1, 2)  # attention.shape = [B,D_seq_len,num_heads,dim//num_heads]
        attention = torch.reshape(attention, (B, D_seq_len, dim))  # attention.shape = [B,D_seq_len,dim]
        attention = self.proj(attention)  # attention.shape = [B,D_seq_len,dim]
        attention = self.proj_drop(attention)  # attention.shape = [B,D_seq_len,dim]
        return attention

class FeedForward(nn.Module):
    """
    前馈神经网络
    用于构建Encoder和Decoder
    """
    def __init__(self,in_feature,hide_feature=None,out_feature=None,dropout_ratio=0):
        super(FeedForward, self).__init__()
        self.hide_feature = hide_feature or in_feature
        self.out_feature = out_feature or in_feature
        self.fc1 = nn.Linear(in_feature, self.hide_feature)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(self.hide_feature,self.out_feature)
        self.dropout = nn.Dropout(dropout_ratio) if dropout_ratio > 0 else nn.Identity()


    def forward(self,x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class PositionEmbeddingLearned(nn.Module):
    """
    可学习的绝对位置编码模块（Learned Absolute Position Embedding）

    与固定正弦编码不同，本模块使用可训练的参数来学习位置编码。
    每个位置（行或列）都有一个独立的可学习向量，通过训练不断优化。

    优点：
    - 灵活性高，可以自适应数据分布
    - 可能学到比固定编码更优的表示

    缺点：
    - 需要额外参数量
    - 受限于预设最大尺寸（50x50），无法外推到更大尺寸
    """

    def __init__(self, num_pos_feats=256):
        """
        初始化可学习位置编码模块

        Args:
            num_pos_feats: 每个方向（行或列）的位置编码维度
                          注意：这里 num_pos_feats 是完整维度（如256），
                          与正弦编码中的 num_pos_feats（一半维度）不同
        """
        super().__init__()

        # 创建行位置嵌入表：50行，每行 num_pos_feats 维
        # nn.Embedding(50, num_pos_feats) 创建一个形状为 [50, num_pos_feats] 的可训练参数矩阵
        # 行索引 0~49 分别对应第0行到第49行的位置编码
        self.row_embed = nn.Embedding(50, num_pos_feats)

        # 创建列位置嵌入表：50列，每列 num_pos_feats 维
        # 列索引 0~49 分别对应第0列到第49列的位置编码
        self.col_embed = nn.Embedding(50, num_pos_feats)

        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        """
        初始化位置编码参数
        使用均匀分布初始化，范围默认在 [0, 1) 之间
        """
        # 对行嵌入表的权重进行均匀分布初始化
        # nn.init.uniform_ 默认范围是 [0, 1)
        nn.init.uniform_(self.row_embed.weight)
        # 对列嵌入表的权重进行均匀分布初始化
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        """
        前向传播：生成可学习的二维位置编码

        Args:
            tensor_list: NestedTensor 对象，包含：
                - tensors: 图像特征图 [batch_size, channels, height, width]
                - mask: 填充掩码 [batch_size, height, width]

        Returns:
            pos: 位置编码 [batch_size, 2*num_pos_feats, height, width]
                 可直接与特征图相加
        """
        # 1. 提取特征图并获取尺寸
        x = tensor_list.tensors  # [B, C, H, W]
        h, w = x.shape[-2:]  # 获取高度 H 和宽度 W

        # 2. 生成列索引和行索引
        # i: 0, 1, 2, ..., w-1  (列索引)
        i = torch.arange(w, device=x.device)  # [w]
        # j: 0, 1, 2, ..., h-1  (行索引)
        j = torch.arange(h, device=x.device)  # [h]

        # 3. 通过嵌入表获取位置编码
        # col_embed(i): 获取第0到w-1列的编码 -> [w, num_pos_feats]
        x_emb = self.col_embed(i)  # [w, num_pos_feats]
        # row_embed(j): 获取第0到h-1行的编码 -> [h, num_pos_feats]
        y_emb = self.row_embed(j)  # [h, num_pos_feats]

        # 4. 扩展并拼接行列编码
        # 目标：生成 [h, w, 2*num_pos_feats] 的完整位置编码矩阵

        # x_emb.unsqueeze(0)      -> [1, w, num_pos_feats]
        # .repeat(h, 1, 1)        -> [h, w, num_pos_feats]
        # 作用：将列编码在行方向重复 h 次，使得每个像素位置都获得该列的编码
        col_emb_expanded = x_emb.unsqueeze(0).repeat(h, 1, 1)  # [h, w, num_pos_feats]

        # y_emb.unsqueeze(1)      -> [h, 1, num_pos_feats]
        # .repeat(1, w, 1)        -> [h, w, num_pos_feats]
        # 作用：将行编码在列方向重复 w 次，使得每个像素位置都获得该行的编码
        row_emb_expanded = y_emb.unsqueeze(1).repeat(1, w, 1)  # [h, w, num_pos_feats]

        # 在最后一个维度（特征维度）上拼接行编码和列编码
        # cat([列编码, 行编码], dim=-1) -> [h, w, 2*num_pos_feats]
        pos = torch.cat([col_emb_expanded, row_emb_expanded], dim=-1)  # [h, w, 2*num_pos_feats]

        # 5. 调整维度顺序并扩展到 batch 维度

        # permute(2, 0, 1)         -> [2*num_pos_feats, h, w]
        # 将特征维度移到最前面，与特征图的 [C, H, W] 格式对齐
        pos = pos.permute(2, 0, 1)  # [2*num_pos_feats, h, w]

        # unsqueeze(0)             -> [1, 2*num_pos_feats, h, w]
        # repeat(x.shape[0], 1, 1, 1) -> [B, 2*num_pos_feats, h, w]
        # 在 batch 维度上复制，使得每个样本都有相同的位置编码
        pos = pos.unsqueeze(0).repeat(x.shape[0], 1, 1, 1)  # [B, 2*num_pos_feats, h, w]

        return pos

