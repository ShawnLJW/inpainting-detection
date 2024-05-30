import torch
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader
from .dataset import ImageDataset


def get_train_loader(args):
    train_ds = ImageDataset("data/genfill", "data/genfill/train.csv")
    train_ds.transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.RandomResizedCrop((256, 256), scale=(0.5, 1.5), interpolation=v2.InterpolationMode.BILINEAR),
            v2.RandomHorizontalFlip(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
    return DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=32, persistent_workers=True)


def get_val_loader(args):
    test_ds = ImageDataset("data/genfill", "data/genfill/test.csv")
    test_ds.transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((256, 256), interpolation=v2.InterpolationMode.BILINEAR),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
    return DataLoader(test_ds, batch_size=128, num_workers=32, persistent_workers=True)
