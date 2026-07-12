from math import inf
import torch
import torch.nn as nn
import config
from dataset import get_dataloder
from tqdm import tqdm
from model import ResNet, Bottleneck
from evaluate import evaluate
import os

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
    model = ResNet(Bottleneck, [3, 4, 23, 3], include_top=True)
    model.to(device)
    model.load_state_dict(torch.load(config.WEIGHT_DIR / "best_acc.pth"))

    # 3.定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 4.定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

    # 5.边训练模型边检验
    best_train_loss = inf
    best_acc = 0
    for epoch in range(config.EPOCHS):
        train_loss =train_one_epoch(model,criterion,optimizer,train_loader,device,epoch)
        print(f"Train Epoch {epoch+1}/{config.EPOCHS} 训练损失: {train_loss:.4f}")
        
        acc =  evaluate(model,val_loader,device)
        print(f"Evaluate Epoch {epoch+1}/{config.EPOCHS} 验证准确率: {acc:.4f}")

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
    

        

    

    


# 训练一个 epoch
def train_one_epoch(model,criterion,optimizer,train_loader,device,epoch):
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

        pbar.set_postfix(loss=total_loss / (batch_idx + 1))

    return total_loss / len(train_loader)

if __name__ == "__main__":
    train()
