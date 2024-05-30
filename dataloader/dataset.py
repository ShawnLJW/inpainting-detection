import os
import pandas as pd
import torch
import torchvision.transforms.v2 as v2
from torchvision.datasets import VisionDataset
from PIL import Image


class ImageDataset(VisionDataset):
    def __init__(self, root, csv_path, transforms=None, img_transforms=None):
        super().__init__(root, transforms)
        self.df = pd.read_csv(csv_path)
        self.df["class"] = self.df["class"].astype(float)
        self.to_pil_image = v2.ToPILImage()
        self.img_transforms = img_transforms

    def __len__(self):
        return len(self.df)

    def loader(self, path):
        with open(os.path.join(self.root, path), "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
            return img

    def mask_loader(self, path):
        with open(os.path.join(self.root, path), "rb") as f:
            img = Image.open(f)
            img = img.convert("L")
            return img

    def __getitem__(self, index):
        img = self.loader(self.df.at[index, "image"])

        target = self.df.at[index, "class"]
        if target == 1:
            mask = self.mask_loader(self.df.at[index, "mask"])
        else:
            w, h = img.size
            mask = self.to_pil_image(torch.zeros((1, h, w)))

        label = self.df.at[index, "class"]
        target = {"masks": mask, "labels": label}
        if self.img_transforms:
            img = self.img_transforms(img)
        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target
