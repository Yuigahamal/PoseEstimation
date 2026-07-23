# config.py
#-------------------------------------
# 基础常量配置
#-------------------------------------

IS_RUN_ON_SERVER = False
ROOT_DIR =  "B:/study/PoseEstimation/VitPose"
DATA_DIR = ROOT_DIR + "/data"
WORK_DIR = ROOT_DIR + "/work_dirs"

COCO_DIR = DATA_DIR + "/coco"
ANNOTATIONS_DIR = DATA_DIR + "/coco/annotations_trainval2017/annotations"
COCO_TRAIN_ANNOTATIONS_PATH = DATA_DIR + "/coco/annotations_trainval2017/annotations/person_keypoints_train2017.json"
COCO_VAL_ANNOTATIONS_PATH = DATA_DIR + "/coco/annotations_trainval2017/annotations/person_keypoints_val2017.json"
COCO_TRAIN_IMAGES_DIR = COCO_DIR + "/train2017"
COCO_VAL_IMAGES_DIR = COCO_DIR + "/val2017"


INPUT_SIZE = (256,192) #输入图像大小
HEATMAP_SIZE = (48,64) #预测热力图大小
HEATMAP_SIGMA = 2 # 热力图sigma值
NUM_KEYPOINTS = 17 # 关键点数量


BATCH_SIZE = 64 if IS_RUN_ON_SERVER else 4
NUM_WORKERS = 8 if IS_RUN_ON_SERVER else 2

BASE_LR = 0.0005 # 基础学习率


#----------------------------
# 模型配置
#----------------------------

 # 定义数据编解码器，用于生成target和对pred进行解码，同时包含了输入图片和输出heatmap尺寸等信息
codec = dict(
    type='MSRAHeatmap',  # 热力图编码器类型
    input_size=INPUT_SIZE,  # 输入图片尺寸
    heatmap_size=HEATMAP_SIZE,  # 输出热力图尺寸
    sigma=HEATMAP_SIGMA  # 热力图sigma值
)

# 模型配置
model = dict(
    type='TopdownPoseEstimator', # 模型结构决定了算法流程

    data_preprocessor=dict( # 数据归一化和通道顺序调整，作为模型的一部分
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),

    backbone=dict(
        type='VIT', # 使用自己写的VIT模型作为backbone
        img_size=INPUT_SIZE,
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
        pos_dropout_ratio=0.1,
    ),
    head=dict(
        type='HeatmapHead', # 使用mmpose中自带的HeatmapHead
        in_channels=768, # 输入通道，即输入序列数据的embed_dim
        out_channels=17, # 输出通道
        deconv_out_channels=(256, 256), # 上采样层输出通道
        deconv_kernel_sizes=(4, 4), # 上采样层卷积核大小

        loss=dict(
            type='KeyPointLoss', # 使用自己写的损失函数
            use_target_weight=True, # 是否使用目标权重
        ),
        decoder=codec # 解码器，用于可视化和后续处理
    ),

    test_cfg=dict(
        flip_test=True, # 是否进行翻转测试
        flip_mode='heatmap', # 翻转模式
        shift_heatmap=False, # 是否进行热力图偏移
    ))


#----------------------------
# 训练配置
#----------------------------

resume = True # 断点恢复
load_from = None # 模型权重加载
# 训练配置
train_cfg = dict(
    max_epochs=210, # 训练轮数
    val_interval=10, # 测试间隔
    by_epoch=True,  # 按 epoch 计数
)

# 优化器配置
optim_wrapper = dict(optimizer=dict(type='AdamW', lr=BASE_LR)) # 优化器和学习率

# 学习率调度配置
param_scheduler = [
    # warmup策略
    dict( type='LinearLR',# 线性预热策略
          begin=0,   # 从第 0 个 iteration 开始
          end=500, # 到第 500 个 iteration 结束
          start_factor=0.001, # 起始学习率 = 初始学习率 * 0.001
          by_epoch=False # 按 iteration 计数（不是 epoch）
          ),

    dict(
        type='CosineAnnealingLR', #  余弦退火
        begin=0, # 从第 0 个 epoch 开始生效
        end=210, # 到第 210 个 epoch 结束
        T_max=210, # 余弦退火的周期长度
        eta_min=1e-6, # 最小学习率
        by_epoch=True # 按 epoch 计数
    )
]



# 根据batch_size自动缩放学习率
auto_scale_lr = dict(base_batch_size=512)

#-----------------------
# 评估配置
#-----------------------

val_evaluator = dict(
    type='CocoMetric', # coco 评测指标
    ann_file=COCO_VAL_ANNOTATIONS_PATH # COCO验证集标注文件路径
)

test_evaluator = val_evaluator # 默认情况下不区分验证集和测试集，用户根据需要来自行定义

val_cfg = dict()
test_cfg = dict()
#----------------------
# 数据集配置
#----------------------

backend_args = dict(backend='local') # 数据加载后端设置，默认从本地硬盘加载
dataset_type = 'CocoDataset' # 数据集类名
data_mode = 'topdown' # 算法结构类型，用于指定标注信息加载策略
data_root = DATA_DIR # 数据存放路径

train_pipeline = [ # 训练时数据增强
    dict(type='LoadImage', backend_args=backend_args), # 加载图片
    dict(type='GetBBoxCenterScale'), # 根据bbox获取center和scale
    dict(type='RandomBBoxTransform'), # 生成随机位移、缩放、旋转变换矩阵
    dict(type='RandomFlip', direction='horizontal'), # 生成随机翻转变换矩阵
    dict(type='RandomHalfBody'), # 随机半身增强
    dict(type='TopdownAffine', input_size=codec['input_size']), # 根据变换矩阵更新目标数据
    dict(
        type='GenerateTarget', # 根据目标数据生成监督信息
        # 监督信息类型
        encoder=codec, # 传入编解码器，用于数据编码，生成特定格式的监督信息
    ),
    dict(type='PackPoseInputs') # 对target进行打包用于训练
]

test_pipeline = [ # 测试时数据增强
    dict(type='LoadImage', backend_args=backend_args), # 加载图片
    dict(type='GetBBoxCenterScale'), # 根据bbox获取center和scale
    dict(type='TopdownAffine', input_size=codec['input_size']), # 根据变换矩阵更新目标数据
    dict(type='PackPoseInputs') # 对target进行打包用于训练
]

train_dataloader = dict( # 训练数据加载
    batch_size=BATCH_SIZE, # 批次大小
    num_workers=NUM_WORKERS, # 数据加载进程数
    persistent_workers=True, # 在不活跃时维持进程不终止，避免反复启动进程的开销
    sampler=dict(type='DefaultSampler', shuffle=True), # 采样策略，打乱数据
    dataset=dict(
            type=dataset_type , # 数据集类名
            data_root=DATA_DIR, # 数据集路径
            data_mode=data_mode, # 算法类型
            ann_file=COCO_TRAIN_ANNOTATIONS_PATH, # 训练集标注文件路径
            data_prefix=dict(img=COCO_TRAIN_IMAGES_DIR), # 训练集图像路径
            pipeline=train_pipeline, # 训练数据流水线
        )
)

val_dataloader = dict(
    batch_size= BATCH_SIZE, # 批次大小
    num_workers=NUM_WORKERS, # 数据加载进程数
    persistent_workers=True, # 在不活跃时维持进程不终止，避免反复启动进程的开销
    drop_last=False, # 是否舍弃最后一个批次
    sampler=dict(type='DefaultSampler', shuffle=False), # 采样策略，不进行打乱
    dataset=dict(
        type=dataset_type , # 数据集类名
        data_root=DATA_DIR, # 数据集路径
        data_mode=data_mode, # 算法类型
        ann_file=COCO_VAL_ANNOTATIONS_PATH, # 标注文件路径
        data_prefix=dict(img=COCO_VAL_IMAGES_DIR), # 图像路径
        test_mode=True, # 测试模式开关
        pipeline=test_pipeline # 数据流水线
    ))

test_dataloader = val_dataloader # 默认情况下不区分验证集和测试集，用户根据需要来自行定义

#------------------
# 通用配置
#------------------
default_scope = 'mmpose'

default_hooks = dict(
    # 迭代时间统计，包括数据耗时和模型耗时
    timer=dict(type='IterTimerHook'),

    # 日志打印间隔，默认每 50 iters 打印一次
    logger=dict(type='LoggerHook', interval=50),

    # 用于调度学习率更新的 Hook
    param_scheduler=dict(type='ParamSchedulerHook'),

    checkpoint=dict(
        # ckpt 保存间隔，最优 ckpt 参考指标。
        # 例如：
        # save_best='coco/AP' 代表以 coco/AP 作为最优指标，对应 CocoMetric 评测器的 AP 指标
        # save_best='PCK' 代表以 PCK 作为最优指标，对应 PCKAccuracy 评测器的 PCK 指标
        # 更多指标请前往 mmpose/evaluation/metrics/
        type='CheckpointHook', # 保存检查点的 Hook
        interval=1,  # 保存检查点的间隔
        save_best='coco/AP', # 保存最优检查点的指标

        # 最优 ckpt 保留规则，greater 代表越大越好，less 代表越小越好
        rule='greater'
    ),

    # 分布式随机种子设置 Hook
    sampler_seed=dict(type='DistSamplerSeedHook'))

env_cfg = dict(
    # cudnn benchmark 开关，用于加速训练，但会增加显存占用
    cudnn_benchmark=False,

    # opencv 多线程配置，用于加速数据加载，但会增加显存占用
    # 默认为 0，代表使用单线程
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),

    # 分布式训练后端设置，支持 nccl 和 gloo
    dist_cfg=dict(backend='nccl')
)

# 可视化器后端设置，默认为本地可视化
vis_backends = [dict(type='LocalVisBackend')]

# 可视化器设置
visualizer = dict(
    type='PoseLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer'
)

log_processor = dict( # 训练日志格式、间隔
    type='LogProcessor', window_size=50, by_epoch=True, num_digits=6)

# 日志记录等级，INFO 代表记录训练日志，WARNING 代表只记录警告信息，ERROR 代表只记录错误信息
log_level = 'INFO'

# 日志和权重保存路径
work_dir = WORK_DIR
