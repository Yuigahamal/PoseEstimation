import torch
from tqdm import tqdm



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