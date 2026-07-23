from pathlib import Path
import os

IS_RUN_ON_SERVER = False
ROOT_DIR = Path(__file__).parent.parent

CONFIG_DIR = ROOT_DIR / "configs"

MODELS_DIR = ROOT_DIR / "models"

DATA_DIR = ROOT_DIR / "data"
COCO_DIR = DATA_DIR / "coco"
ANNOTATIONS_DIR = COCO_DIR / "annotations_trainval2017/annotations"
COCO_TRAIN_ANNOTATIONS_PATH = ANNOTATIONS_DIR / "person_keypoints_train2017.json"
COCO_VAL_ANNOTATIONS_PATH = ANNOTATIONS_DIR / "person_keypoints_val2017.json"
COCO_TRAIN_IMAGES_DIR = COCO_DIR / "train2017" / "train2017"
COCO_VAL_IMAGES_DIR = COCO_DIR / "val2017" / "val2017"


DATASETS_DIR = ROOT_DIR / "datasets"

ENGINE_DIR = ROOT_DIR / "engine"

WORK_DIR = ROOT_DIR / "work_dirs"

UTILS_DIR = ROOT_DIR / "utils"


INPUT_SIZE = (192,256) #输入图像大小
HEATMAP_SIZE = (48,64) #预测热力图大小
HEATMAP_SIGMA = 2 # 热力图sigma值
NUM_KEYPOINTS = 17 # 关键点数量


BATCH_SIZE = 64 if IS_RUN_ON_SERVER else 4
NUM_WORKERS = 8 if IS_RUN_ON_SERVER else 2

BASE_LR = 0.0005 # 基础学习率


# 设置环境变量
def set_env():
    os.environ["ROOT_DIR"] = str(ROOT_DIR)
    os.environ["CONFIG_DIR"] =str(CONFIG_DIR)
    os.environ["MODELS_DIR"] = str(MODELS_DIR)
    os.environ["DATA_DIR"] = str(DATA_DIR)
    os.environ["COCO_DIR"] = str(COCO_DIR)
    os.environ["ANNOTATIONS_DIR"] = str(ANNOTATIONS_DIR)

    os.environ["COCO_TRAIN_ANNOTATIONS_PATH"] = str(COCO_TRAIN_ANNOTATIONS_PATH)

    os.environ["COCO_VAL_ANNOTATIONS_PATH"] = str(COCO_VAL_ANNOTATIONS_PATH)

    os.environ["COCO_TRAIN_IMAGES_DIR"] = str(COCO_TRAIN_IMAGES_DIR)

    os.environ["COCO_VAL_IMAGES_DIR"] = str(COCO_VAL_IMAGES_DIR)

    os.environ["DATASETS_DIR"] = str(DATASETS_DIR)

    os.environ["ENGINE_DIR"] = str(ENGINE_DIR)

    os.environ["WORK_DIR"] = str(WORK_DIR)

    os.environ["UTILS_DIR"] = str(UTILS_DIR)

    os.environ["INPUT_SIZE"] = str(INPUT_SIZE)

    os.environ["HEATMAP_SIZE"] = str(HEATMAP_SIZE)

    os.environ["HEATMAP_SIGMA"] = str(HEATMAP_SIGMA)
    os.environ["NUM_KEYPOINTS"] = str(NUM_KEYPOINTS)
    os.environ["BATCH_SIZE"] = str(BATCH_SIZE)
    os.environ["NUM_WORKERS"] = str(NUM_WORKERS)

    os.environ["BASE_LR"] = str(BASE_LR)




