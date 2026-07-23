from math import inf
import torch
import torch.nn as nn

import config
from dataset import get_dataloder
from tqdm import tqdm
from model import VIT_Base
from evaluate import evaluate
import pathlib
import os
from torch.optim.lr_scheduler import CosineAnnealingLR,LinearLR

def train():

    # 1.加载数据集
    print("开始获取数据集")
    train_loader = get_dataloder(is_train=True,fraction = config.DATA_FRACTION)
    print("获取训练数据集完成")
    val_loader = get_dataloder(is_train=False)
    print("获取验证数据集完成")

    # 2.设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"训练设备为: {device_name}")

    # 2.加载模型
    model = VIT_Base(img_size=config.IMG_SIZE, patch_size=config.PATCH_SIZE, num_class=config.NUM_CLASSES)
    model.to(device)
    if os.path.exists(config.WEIGHT_DIR / "best_acc.pth"):
        model.load_state_dict(torch.load(config.WEIGHT_DIR / "best_acc.pth"))

    # 3.定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 4.定义优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.BASE_LR)
    # 普通余弦退火（一个周期，从初始值逐渐降到 0）
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [
            LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=config.WARMUP_EPOCHS), # 前几轮Warmup
            CosineAnnealingLR(optimizer, T_max=config.EPOCHS -  config.WARMUP_EPOCHS, eta_min=config.ETA_MIN)
        ]
        ,
        [config.WARMUP_EPOCHS]
    )

    # 5.打开日志文件
    if not os.path.exists(config.LOGS_DIR):
        os.mkdir(config.LOGS_DIR)
    if not os.path.exists(config.LOGS_DIR/ "train.log"): # 或者 pathlib.Path.exist
        pathlib.Path.touch(config.LOGS_DIR / "train.log")
    f = open(config.LOGS_DIR / "train.log", "a")

    # 6.边训练模型边检验
    best_train_loss = inf
    best_acc = 0
    for epoch in range(config.EPOCHS):
        train_loss =train_one_epoch(model,criterion,optimizer,scheduler,train_loader,device,epoch)
        print(f"Train Epoch {epoch+1}/{config.EPOCHS} 训练损失: {train_loss:.4f}")
        
        acc =  evaluate(model,val_loader,device)
        print(f"Evaluate Epoch {epoch+1}/{config.EPOCHS} 验证准确率: {acc:.4f}")

        write_log(f,epoch, train_loss, acc) # 写日志

        if acc > best_acc:
            best_acc = acc
            if not os.path.exists(config.WEIGHT_DIR) :
                os.mkdir(config.WEIGHT_DIR)
            torch.save(model.state_dict(), config.WEIGHT_DIR / "best_acc.pth")

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            if not os.path.exists(config.WEIGHT_DIR) :
                os.mkdir(config.WEIGHT_DIR)
            torch.save(model.state_dict(), config.WEIGHT_DIR / "best_train_loss.pth")

    # 7.关闭日志文件
    f.close()

# 训练一个 epoch
def train_one_epoch(model,criterion,optimizer,scheduler,train_loader,device,epoch):
    model.train()

    total_loss = 0.0
    pbar = tqdm(train_loader,
    desc=f"Train Epoch {epoch+1}/{config.EPOCHS}",
    leave=True # 每个 epoch 结束后保留进度条
    )

    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        pbar.set_postfix(loss=total_loss / (batch_idx + 1),lr = optimizer.param_groups[0]['lr'])
        scheduler.step()

    return total_loss / len(train_loader)

# 写日志
def write_log(f,epoch, train_loss, acc):
    f.write(f"Epoch {epoch+1}/{config.EPOCHS} 训练损失: {train_loss:.4f} 验证准确率: {acc:.4f}\n")


if __name__ == "__main__":
    train()
