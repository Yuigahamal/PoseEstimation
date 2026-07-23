from pathlib import Path

IS_RUN_ON_SERVER = False # 是否在云上跑

ROOT_DIR = Path(__file__).parent.parent # 项目根目录

DATA_DIR = ROOT_DIR / "data"   # 数据集路径

WEIGHT_DIR = ROOT_DIR / "weight"  # 模型权重路径

LOGS_DIR  = ROOT_DIR / "logs" # 日志路径


NUM_CLASSES = 10  # 类别数
PATCH_SIZE  = 4 # 划分块的大小
IMG_SIZE  = 32 # 输入图片大小
DATA_FRACTION = 1.0 if IS_RUN_ON_SERVER else 0.01  # 训练数据量，1.0 表示全量
BATCH_SIZE = 128 if IS_RUN_ON_SERVER else 16  # 批次大小
NUM_WORKERS = 64 if IS_RUN_ON_SERVER else 8   # 工作线程数
EPOCHS = 120 if IS_RUN_ON_SERVER  else 10     # 训练轮数
WARMUP_EPOCHS = 10  # 预热轮数
BASE_LR = 1e-3  # 基础学习率
ETA_MIN = 1e-6 # 最小学习率
WEIGHT_DECAY = 0.05 # 权重衰减




