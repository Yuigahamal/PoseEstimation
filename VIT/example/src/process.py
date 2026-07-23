# 处理原始数据，生成处理后的数据文件(json,exl,txt等)
import os
import config


def process():
    print("开始处理数据")
    if not os.path.exists(config.DATA_DIR):
        os.mkdir(config.DATA_DIR)
    print("处理完成")


if __name__ == "__main__":
    process()

    
