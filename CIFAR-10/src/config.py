from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

DATA_DIR = ROOT_DIR / "data"   # 数据集路径

WEIGHT_DIR = ROOT_DIR / "weight"  # 模型权重路径

LOGS_DIR  = ROOT_DIR / "logs" # 日志路径


NUM_CLASSES = 10 # 类别数
DATA_FRACTION = 1.0  # 训练数据量，1.0 表示全量
BATCH_SIZE = 16 # 批次大小
NUM_WORKERS = 8 # 工作线程数
EPOCHS = 10 # 训练轮数
LR = 0.005 # 学习率




