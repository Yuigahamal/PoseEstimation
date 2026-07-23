
import torch
import torch.nn as nn
class BasicModule(nn.Module):
    

    """
    BasicModule类是ResNet网络中的一个基本构建块，用于构建深层网络。
    它实现了残差连接，通过两个3x3卷积和一个1x1卷积的组合来提取特征。
    用于构建ResNet-18和ResNet-34网络
    """

    expansion = 1 # 该模块输出通道相比输入通道的扩大倍数
    def __init__(self, in_channels, out_channel,stride = 1, downsample=None):
        super(BasicModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)

        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
 
        self.downsample = downsample # 下采样映射函数（通常也是一个包含 1x1 卷积的序列）。当输入特征图与输出特征图的尺寸或通道数不一致时，需要通过 downsample 调整输入特征图，以便两者能够相加。

        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):

    """
    Bottleneck类是ResNet网络中的一个基本构建块，用于构建深层网络。
    它实现了残差连接，通过两个3x3卷积和一个1x1卷积的组合来提取特征。
    用于构建ResNet-50、ResNet-101和ResNet-152网络
    """
    expansion = 4 # 该模块输出通道相比输入通道的扩大倍数
    def __init__(self, in_channel, out_channel,stride = 1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)

        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)    
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.conv3 = nn.Conv2d(out_channel, out_channel*Bottleneck.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel*Bottleneck.expansion)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample # 下采样映射函数（通常也是一个包含 1x1 卷积的序列）。当输入特征图与输出特征图的尺寸或通道数不一致时，需要通过 downsample 调整输入特征图，以便两者能够相加。



    def forward(self,x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    """
    ResNet类是ResNet网络的主类，用于构建深层网络。
    它包含了多个BasicModule或Bottleneck模块，通过堆叠这些模块来构建深层网络。
    用于构建ResNet-18、ResNet-34、ResNet-50、ResNet-101和ResNet-152网络
    
    """

    def __init__(self, block, num_blocks, num_classes=10,include_top = False):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        # self.conv1 = nn.Conv2d(3,self.in_channel,kernel_size=7,stride =2,padding =3,bias =False)
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=3, stride=1, padding=1, bias=False) # 适配CIFAR-10的32尺寸
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.Identity()  #不做任何操作，适配CIFAR-10的32尺寸

        self.layer1 = self.__make__layer(block, 64, num_blocks[0])
        self.layer2 = self.__make__layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self.__make__layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self.__make__layer(block, 512, num_blocks[3], stride=2)

        self.relu = nn.ReLU(inplace=True)

        if self.include_top:  # 是否包含下游任务，如分类等
            self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1)) # 全局平均池化
            self.fc = nn.Linear(512 * block.expansion, num_classes) # 输出层，用于分类

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') # 初始化卷积层的权重


    def __make__layer(self,block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != block.expansion * channel:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, block.expansion * channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(block.expansion * channel)
            )
        
        layers = []

        layers.append(block(self.in_channel, channel, stride, downsample))  # layer中的第一个block与其他block不同
        # 第一个block要下采样，而且会有downsample
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)
    

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def resnet101(num_classes=10,top_include = False):
    return ResNet(Bottleneck, [3, 4, 23, 3],num_classes,top_include)

def resnet152(num_classes=10,top_include = False):
    return ResNet(Bottleneck, [3, 8, 36, 3],num_classes,top_include)

def resnet50(num_classes=10,top_include = False):
    return ResNet(Bottleneck, [3, 4, 6, 3],num_classes, top_include)

def resnet34(num_classes=10,top_include = False):
    return ResNet(BasicModule, [3, 4, 6, 3],num_classes,top_include)

def resnet18(num_classes=10,top_include = False):
    return ResNet(BasicModule, [2, 2, 2, 2],num_classes,top_include)



if __name__ == "__main__":
    resnet18 = resnet18(num_classes=10,top_include = True)
    resnet34 = resnet34(num_classes=10,top_include = True)
    resnet50 = resnet50(num_classes=10,top_include = True)
    resnet101 = resnet101(num_classes=10)
    resnet152 = resnet152(num_classes=10)
    
    x= torch.randn(1,3,32,32)
    print(resnet18(x).shape)
    print(resnet34(x).shape)
    print(resnet50(x).shape)
    print(resnet101(x).shape)
    print(resnet152(x).shape)