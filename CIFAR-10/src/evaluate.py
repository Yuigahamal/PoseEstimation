import torch
from tqdm import tqdm
from model import resnet50
import dataset
import config
import os
def run_evaluate():
    # 1.获取验证集
    val_dataloder =  dataset.get_dataloder(is_train=False)


    # 2.获取设备

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"训练设备为: {device_name}")

    # 3.加载模型
    model = resnet50().to(device)
    if os.path.exists(config.WEIGHT_DIR / "best_acc.pth"):
        model.load_state_dict(torch.load(config.WEIGHT_DIR / "best_acc.pth"))

    # 4.评估模型
    acc = evaluate(model, val_dataloder, device)
    print(f"模型在验证集上的准确率为: {acc:.4f}")

# 图像分类评估函数,返回准确率
def evaluate(model, val_loader, device):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Evaluate", leave=True)
        for images, labels in pbar:
            images = images.to(device) # (B,C,W,H)
            labels = labels.to(device) # (B,1)

            outputs = model(images) # (B,10)
            predictions = outputs.argmax(dim=1)

            total += labels.size(0)
            correct += (predictions == labels).sum().item()
            pbar.set_postfix(acc=correct / total)

    return correct / total

if __name__ == '__main__':
    run_evaluate()