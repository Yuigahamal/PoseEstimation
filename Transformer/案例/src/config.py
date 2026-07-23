from pathlib import Path


# 路径配置
ROOT_DIR = Path(__file__).parent.parent

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"


# 模型参数

EMBEDDING_DIM = 128  # 词向量维度大小
DIM_MODEL = EMBEDDING_DIM  # 编码器的维度大小
N_HEAD =  4 # 多头注意力的头数
NUM_ENCODER_LAYER = 2 # 编码器层数
NUM_DECODER_LAYER = 2 # 解码器层数
DIM_FEEDFORWARD = 512 # 前馈网络的维度大小




# 训练参数
BATCH_SIZE = 64
NUM_WORKERS = 8
LEARNING_RATE = 0.001
NUM_EPOCHS = 30

MAX_SEQ_LENGTH = 128