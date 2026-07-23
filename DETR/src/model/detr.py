import torch
import torch.nn as nn
import torchinfo
from torchvision.models import resnet50


class DETR(nn.Module):
    def __init__(self,num_classes = 10,hidden_dim = 256,attn_head=8,num_encoder_layers=6,num_decoder_layers=6):
        super(DETR,self).__init__()
        self.backbone = nn.Sequential(*list(resnet50(weights='ResNet50_Weights.IMAGENET1K_V1').children())[:-2]) # 下采样32倍
        self.conv = nn.Conv2d(2048,hidden_dim,kernel_size=1)
        self.transformer = nn.Transformer(d_model=hidden_dim, nhead=attn_head, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,batch_first=True)

        # 预测头
        self.linear_class =  nn.Linear(hidden_dim,num_classes+1) # 分类头,加1个背景类
        self.linear_bbox  =  nn.Linear(hidden_dim,4) # 边界框回归头, 4个坐标

        # 查询向量
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim)) # query_pos 是 100 个可学习的位置编码向量
        # 可以把这 100 个queries看作模型提出的 100 个"问题"："第1个位置有没有物体？是什么？框在哪里？"每个query最终输出一个预测结果。

        # 位置编码,表示最大支持 50 行和 50 列的位置索引（即支持最大 50×50 的特征图）
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2)) # [50, hidden_dim // 2]
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2)) # [50, hidden_dim // 2]



    # 生成位置编码
    def __make_pos_encoding(self,h,w):
        pos = torch.concat([
            self.col_embed[:w].unsqueeze(0).repeat(h,1,1), # [50,hidden_dim // 2] -> [w,hidden_dim // 2]-> [1,w,hidden_dim] -> [h,w,hidden_dim]
            self.row_embed[:h].unsqueeze(1).repeat(1,w,1), # [50,hidden_dim // 2] -> [h,hidden_dim // 2]-> [h,1,hidden_dim] -> [h,w,hidden_dim]
        ],dim=-1) # pos.shape = [h,w,hidden_dim]
        pos = torch.flatten(pos,0,1) # pos.shape = [h*w=seq_len, hidden_dim]
        pos = torch.unsqueeze(pos,0) # pos.shape = [1, seq_len, hidden_dim]
        return pos


    def forward(self,x):
        # x.shape = [batch_size, 3,W,H]
        x = self.backbone(x) # x.shape = [batch_size, 2048, W/32, H/32]
        x = self.conv(x) # x.shape = [batch_size, hidden_dim, W/32, H/32]
        batch_size,dim,w,h = x.shape
        x = torch.reshape(x,(batch_size,dim,w*h)) # x.shape = [batch_size, hidden_dim, w*h = seq_len]
        x = torch.transpose(x,1,2) # x.shape = [batch_size, seq_len, hidden_dim]
        pos = self.__make_pos_encoding(h,w) # pos.shape = [1, seq_len, hidden_dim]
        x = x+pos


        out = self.transformer(x,self.query_pos.unsqueeze(0)) # out.shape = [batch_size, 100, hidden_dim]

        out_class = self.linear_class(out) # out_class.shape = [batch_size, seq_len, num_classes+1]
        out_bbox  = self.linear_bbox (out) # out_bbox .shape = [batch_size, seq_len, 4]
        out_bbox = torch.sigmoid(out_bbox) # 输出归一化的坐标预测
        return out_class,out_bbox






if __name__ == '__main__':
    model = DETR()
    input = torch.randn(1, 3, 224, 224)
    out_class,out_bbox = model(input)
    print(out_class.shape,out_bbox.shape)

    info = torchinfo.summary(model, input_size=(1, 3, 224, 224),verbose=0)
    print(info)

