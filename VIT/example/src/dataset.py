from torchvision import transforms,datasets
import config
from torch.utils.data import DataLoader, Subset

def get_transform(is_train=True) -> transforms.Compose:
    if is_train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4), # 先填充到40px,再随机裁剪到32px
            transforms.RandomHorizontalFlip(), # 随机水平翻转
            transforms.ToTensor(), # 转换为张量
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # 标准化
        ])
    else:
        return transforms.Compose([
            transforms.Resize(32), # 裁剪到32px
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

def get_dataloder(is_train=True, fraction=1.0) -> DataLoader:
    download_root  =  config.DATA_DIR
    transform = get_transform(is_train)
    dataset = datasets.CIFAR10(
        root=download_root,
        train=is_train,
        download=True,
        transform=transform,
    )

    if not 0 < fraction <= 1:
        raise ValueError("fraction 必须在 (0, 1] 范围内")

    if fraction < 1:
        subset_size = max(1, int(len(dataset) * fraction))
        dataset = Subset(dataset, range(subset_size))

    print(f"数据集长度为: {len(dataset)}")

    dataloder = DataLoader(dataset, config.BATCH_SIZE, shuffle=is_train, num_workers=config.NUM_WORKERS)
    return dataloder

if __name__ == "__main__":
    train_loader = get_dataloder(is_train=True, fraction=config.DATA_FRACTION)
    val_loader = get_dataloder(is_train=False, fraction=config.DATA_FRACTION)
