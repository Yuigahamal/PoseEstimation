import argparse
from pathlib import Path

import torch
from PIL import Image

import config
from dataset import get_transform
from model import Bottleneck, ResNet


CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def build_model():
    return ResNet(Bottleneck, [3, 4, 23, 3], include_top=True)


def load_model(weights_path, device):
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"找不到权重文件: {weights_path}")

    model = build_model().to(device)
    checkpoint = torch.load(weights_path, map_location=device)

    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
    else:
        state_dict = checkpoint

    if state_dict and next(iter(state_dict)).startswith("module."):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict(model, image_path, device):
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"找不到图片文件: {image_path}")

    transform = get_transform(is_train=False)
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        confidence, class_index = torch.max(probabilities, dim=0)

    return class_index.item(), confidence.item()


def parse_args():
    parser = argparse.ArgumentParser(description="使用 ResNet101 预测单张 CIFAR-10 图片类别")
    parser.add_argument("image", type=Path, help="待预测图片路径")
    parser.add_argument(
        "--weights",
        type=Path,
        default=config.MODEL_DIR / "resnet101_cifar10.pth",
        help="训练好的模型权重路径",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"预测设备为: {device_name}")

    model = load_model(args.weights, device)
    class_index, confidence = predict(model, args.image, device)
    class_name = CIFAR10_CLASSES[class_index]

    print(f"图片路径: {args.image}")
    print(f"预测类别索引: {class_index}")
    print(f"预测类别名称: {class_name}")
    print(f"置信度: {confidence:.4%}")


if __name__ == "__main__":
    main()
