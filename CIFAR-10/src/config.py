from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

DATA_DIR = ROOT_DIR / "data"  

WEIGHT_DIR = ROOT_DIR / "weight"

LOGS_DIR  = ROOT_DIR / "logs"

DATA_FRACTION = 1.0  # 训练数据量，1.0 表示全量
BATCH_SIZE = 16 # 批次大小
NUM_WORKERS = 8 # 工作线程数
EPOCHS = 10 # 训练轮数
LR = 0.001 # 学习率




