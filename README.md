# PoseEstimation

体姿势估计旨在定位输入数据(例如图像、视频 或信号)中的人体解剖关键点或身体部位。它是使机器能够深入理解人类行为的关键组成部分,并已成为计算机视觉及相关领域的一个突出问题。
现阶段优化方法: **网络架构设计**、**网络训练细化**和**后处理**。


## 1. 网络架构设计方法

 对于PoseEstimation，网络架构设计的方法通常采用两种通用框架:**自上而下的框架**和**自下而上的框架**。


 ### 1.1 自上而下范式
 自上而下的范式通常先检测人体边界框，然后对每个边界框执行单人姿势估计。自上而下的范式可以进一步划分为
 **基于回归的**、**基于热力图的**、**基于视频的**和**基于模型压缩的**方法。

自顶向下框架的架构包括以下关键组件:用于生成人体边界框的对象检测器(如YOLO)和用于检测人体关键点位置的姿态估计器(如HRNet)。目标检测器决定了人体提案检测的性能, 并进一步影响姿势估计。另一方面,姿态检测器是框架的核心,直接决定姿态估计的准确性。总之,自上而下的框架具有高度可扩展性,可以随着对象检测器和姿势检测器的进步而不断改进。

如下图所示:
<img src="resource/images/Top-Down Paradigm.png" alt="Top-Down Methods"  style="display: block; margin: 0 auto;">

(SPPE是Single-Person Pose Estimator，即单人姿势估计器)

 #### 1.1.1 基于回归的方法 
 基于回归的方法**最开始尝试直接通过端到端的网络学习直接回归关键点的坐标位置。**


 ##### 1.1.1.1 基本坐标回归
 例如DeepPose就是一种基于坐标回归方法的网络架构。它以AlexNet作为backbone提取图像特征，然后使用全连接层回归关键点坐标。 
 例如输入张量为$(batch,C,H,W)$,若有$K$个关键点，则输出张量为$(batch,2*K)$。这$2*K$个数值按顺序排列，就是 $[x_1, y_1, x_2, y_2, ..., x_K, y_K]$,表示$K$个关键点的坐标。

 ##### 1.1.1.2 结构感知回归
 接着有学者提出了一种也是基于回归的自校正模型(self-correcting model),可以称为结构感知回归。


 #### 1.1.2 基于热力图的方法
基于热力图的方法是目前广泛采用的。
这种方法不直接预测关键点坐标，而是预测每个关键点在图像中的概率分布。(注：姿态估计中的 Heatmap 严格来说并不是概率分布（Probability Distribution），而是概率密度响应图（Confidence Map / Likelihood Map）。论文中习惯称它为 Heatmap，但它通常没有经过归一化，因此所有值的和并不等于1。)
例如输入张量为$[batch, C, H, W]$,若有$K$个关键点，则输出张量为$[batch, K, H_m, W_m]$。这张概率图上每个像素的值表示“这个像素是关节”的概率。$H_m$和$W_m$是热力图的高度和宽度，通常比原图小。

疑问:为什么热力图的尺寸比原图小?
答:因为热力图的尺寸比原图小，所以可以减少计算量。例如原图尺寸为$256*192$，热力图尺寸为$64*48$，可以看到$64/256=1/4$，$48/192=1/4$，热力图的尺寸是原图的$1/4$。说明Heatmap上的一个像素对应原图上的4×4个像素区域。若Heatmap预测$(30,21)$点某个关键点概率为$0.9$,则原图上的对应位置$(30*4,21*4)=(120,84)$的概率为$0.9$。实际上，大多数论文不会直接乘4还原到原图坐标点，因为这样会产生**量化误差（Quantization Error）**。


热力图(Heatmap)通常直接由二维高斯函数生成:
$$
H(x,y) = \exp\left(-\frac{(x-x_0)^2 + (y-y_0)^2}{2\sigma^2}\right)
$$

其中：

- $H(x,y)$：热力图中坐标$(x,y)$的值。
- $(x_0,y_0)$：热力图中的关键点（Ground Truth）的峰值坐标。峰值坐标由原图中关键点的坐标除以原图与热力图的缩放比例得到。例如原图大小为$256 × 256$,热力图大小为$64 × 64$,缩放比例为4，如果一个关键点在原图中的坐标为$(100, 120)$,则该关键点在热力图中对应的峰值坐标为$（25,30）$,即$x_0 = 25 , y_0 =30$
- $\sigma$：高斯核的标准差，控制热力图扩散范围。**$\sigma$（标准差）没有固定值，而是一个超参数（Hyperparameter）**，需要根据**热力图分辨率**和**数据集**来选择。但通常热力图都会在数据集中给出了，并不自己要手动去设置。
- $(x,y)$：热力图上的任意一个像素坐标。

注意这里**没有任何归一化操作**。

因此：

- 峰值固定为1；
- 离中心越远值越小；
- **并不要求所有值加起来等于1。**

例如：

0 0.1 0.2

0.4 0.8 1.0

0.3 0.6 0.9

代表性的基于热力图方法的架构有 **Iterative Architecture（迭代架构）、Symmetric Architecture（对称架构）、Asymmetric Architecture（对称架构）、High Resolution Architecture（高分辨率架构）**。

##### 1.1.2.1 Iterative Architecture（迭代架构）
它的核心思想可以概括为一句话：**第一次预测只是一个粗略结果，然后不断利用前一次预测结果进行修正（Refinement），最终得到更加准确的关键点位置。**

这是对“迭代架构”核心思想的概括。它不像现代网络“一步到位”地预测关键点，而是采用**“粗估计 → 反复修正”**的策略。首先生成一张粗糙的热图（初步猜测关节在哪），然后通过多个阶段（Stage）逐步优化，让热图越来越精确。

**Ramakrishna V, Munoz D, Hebert M, Bagnell JA,  Sheikh Y (2014) Pose machines: Articulated pose estimation via inference machines.**是早期代表作。它使用多个串联的预测模块，后一个模块会参考前一个模块的预测结果，从而在空间上逐步“聚焦”到正确的关节位置。

**Wei SE, Ramakrishna V, Kanade T, Sheikh Y (2016)  Convolutional pose machines. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)** 在上文基础上做了关键升级：用卷积神经网络（CNN）替换了之前的模型。更重要的是，CPM(卷积姿势器)通过增大感受野（Receptive Field），让网络在预测手腕时，能“看到”肩膀甚至胯部的位置，从而隐式地学习了人体骨架的几何约束（比如手腕通常不会长在膝盖上）。
CMP还提出了中间监督来缓解迭代架构中梯度消失的固有问题。这是CPM的一个关键技术。因为网络有多个阶段（很深），梯度在反向传播时容易衰减（Vanishing Gradient），导致浅层网络学不到东西。中间监督是指在每个阶段的输出端都计算一次损失（Loss），强制让每一层都接收到有效的梯度信号，确保每一阶段都在“认真工作”。
具体来说中间监督”指的是 CPM 里的 **stage-wise auxiliary loss（阶段式辅助损失）**，不是只监督最后输出，而是每个阶段都单独加损失。
具体做法是:
- 每一阶段都会输出一组 **belief maps**，也就是每个关节的热图预测。
- 用由真实关节位置生成的 Gaussian 目标热图作为监督信号。
- 对每个阶段的预测热图和目标热图计算 L2 loss。
- 把所有阶段的损失相加，作为总损失一起端到端训练。

CMP的论文中提到用于实现这个**stage-wise auxiliary loss（阶段式辅助损失）**的具体落地实现方案是在每个 stage 后面都有一个 **intermediate loss layer（中间损失层）**，但要注意，它本身不是一个可学习的网络层，而是每个 stage 后面挂的一个监督分支。

多阶段具体网络结构如下图所示：
<img src="resource/images/Iterative Architecture.png" alt="Iterative Architecture"  style="display: block; margin: 0 auto;">
- **Feature Maps**：指的是卷积网络提取出来的中间特征图，也就是 image encoder / CNN backbone 输出的空间特征表示，不是标签。
- **HeatMap**：图里这个通常指的是模型当前阶段预测出来的 heatmap / belief maps，也就是上一阶段输出、传给下一阶段继续细化的那一份。

**stage-wise auxiliary loss** 具体网络如下图所示：
<img src="resource/images/stage-wise auxiliary loss.png" alt="stage-wise auxiliary loss"  style="display: block; margin: 0 auto;">

但尽管**stage-wise auxiliary loss（阶段式辅助损失）** 缓解了多阶段模型的梯度消失问题，但每个阶段仍然无法构建深层子网络来提取有效的语义特征，这极大地限制了它们的拟合能力。

但随着ResNet出现，它允许梯度通过“高速公路”直接传回浅层，从而支持训练上百层的深度网络。有了深度，网络就能提取更丰富、更抽象的语义特征。受益于这种方式，许多大型模型被设计出来，极大地促进了二维HPE的进程。

##### 1.1.2.2 Symmetric Architecture（对称架构）
对称架构深度模型通常采用从高到低(下采样，Encoder)和从低到高（上采样，Decoder）的框架，类似于**UNet**。
例如**Newell A, Yang K, Deng J (2016) Stacked hourglass  networks for human pose estimation. In: European conference on computer vision, Springer** 提出一种 **Stacked Hourglass Network (堆叠沙漏网络)**
简单来说一个 **hourglass** 就类似一个小的**UNet**
- 先把图像特征不断下采样，获得大范围上下文信息  
- 再上采样回去，恢复关节定位所需的空间精度  
- 中间通过跳连把浅层细节和深层语义融合起来  
- 一个 **hourglass** 结束后，还可以再接一个 **hourglass** 继续 refine

一个 **hourglass** 的结构图就如下图所示:

**Stacked** 表示不是一个 **hourglass** 就结束，而是多个 **hourglass** 串起来：
<img src="resource/images/stacked hourglass.png" alt="stacked hourglass"  style="display: block; margin: 0 auto;">

- 第 1 个 hourglass 先粗略找到关节
- 第 2 个 hourglass 继续修正
- 第 3 个 hourglass 再进一步细化

所以它本质上是一个逐步 **refinement** 的过程。

##### 1.1.2.3 Asymmetric Architecture(非对称结构)
非对称性架构是指`high-to-low process is heavy` 和 `low-to-high process is light`。
`high-to-low process is heavy`：从高分辨率图像到低分辨率特征图这一步很重，通常用 VGGNet、ResNet 这类分类网络 backbone，负责提取强语义特征。
`low-to-high process is light`：从低分辨率特征恢复到高分辨率 heatmap 这一步比较轻，只用少量上采样或转置卷积。

**Cascaded Pyramid Network for Multi-Person Pose Estimation** 提出的 **CPN（Cascaded Pyramid Network，级联金字塔网络）** 就是一个典型的非对称架构。
其模型的架构如下图所示：

<img src="resource/images/Cascaded Pyramid Network.png" alt="Cascaded Pyramid Network"  style="display: block; margin: 0 auto;">
GlobalNet是第一部分，这个部分先用 ResNet 作为backbone,从输入图像中图区出不同层级的特征图，然后通过FPN（特征金字塔）进行特征融合,最后输出组略热图,如下图所示：
<img src="resource/images/FPN.png" alt="FPN"  style="display: block;width:300px;height:1000px; margin: 0 auto;" >
- `C2, C3, C4, C5`：来自 ResNet 不同阶段的特征。
- `C5`：分辨率最低，但人体整体语义最强，比如能判断“这是一个人、身体大概朝向如何”。
- `C2`：分辨率最高，保留更多边缘、轮廓、局部细节。
- `1x1 Conv`：把不同层的通道数统一，方便后面相加。
- `Upsample`：把深层低分辨率特征放大到上一层的分辨率。通常采用双线性插值。
- `Add`：把上采样后的深层语义特征和当前层的浅层空间特征融合。
- `P2, P3, P4, P5`：融合后的金字塔特征。为后面的 RefineNet 提供多层特征

Re



##### 1.1.2.4 High Resolution Architecture(高分辨率架构)
高分辨率架构与以前的模型不同，最具代表的高分辨率模型是**HRNet**。它能在整个过程中保持高分辨率表示，并进行重复的多尺度融合，每个分辨率特征都从所有分辨率接收丰富的信息。
后来广泛的研究都是基于**HRNet**为**backbone**,进一步结合门控机制和特征注意模块来选择和融合判别性和 注意感知特征。



 #### 1.1.3 基于视频的方法
大多数现在有方法在静态图像上进行训练的。直接将基于图像的模型应用于视频(图像序列)，可能会导致不令人满意的结果,因为它们未能 考虑视频帧之间的时间一致性。为了克服这一困境, 许多方法已经探索利用**附加时间信息**来实现更高的姿态检测精度。
如何利用时间这个信息呢，目前主要有四种方法:**基于光流的、基于RNN的、基于姿势跟踪的和基于关键帧的范式**
 ##### 1.1.3.1 基于光流的范式
 ##### 1.1.3.2 基于RNN的范式
 LSTM姿态机
### 1.2 自下而上的范式
自下而上和自上而下框架之间的主要差异在于是**否采用人体检测器来检测人体边界框**。
自下而上的方法不依赖于人类检测,直接在原 始图像中进行关键点估计。,从而减少了计算开销。然 而,这个过程提出了一个新的挑战:如何判断估计关 节的身份？
根据确定关键点身份的方式，我们将自下而上的发放分为**基于人类中心回归的(human center regression-based)、基于关联嵌入的(associate embeddingbased)和基于部分字段(part field-based)的方法** 。

#### 1.2.1 基于人类中心回归的
基于人类中心回归的方法通常采用**回归方法**来预测人体的中心位置，然后根据中心位置和关键点位置之间的关系来预测关键点位置。这种方法的关键在于如何定义和计算中心位置和关键点位置之间的关系。常见的计算方法包括使用关键点之间的距离、角度等几何特征，或者使用关键点之间的相对位置关系。这些方法通常需要大量的训练数据，并且对训练数据的标注质量要求较高。

#### 1.2.2 基于关联嵌入的




