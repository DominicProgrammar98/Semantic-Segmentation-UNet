import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob
import numpy as np

data_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

class CustomDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

        self.img_list = glob.glob(os.path.join(images_dir, '*.jpg'))
        self.mask_list = glob.glob(os.path.join(masks_dir, '*.png'))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_path = self.img_list[index]
        mask_path = self.mask_list[index]

        img = Image.open(img_path)
        mask = Image.open(mask_path)

        

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)

        return img, mask


images_dir = '/home/rteam1/bazazpour/Semantic-Segmentation-UNet/dataset/images'
masks_dir = '/home/rteam1/bazazpour/Semantic-Segmentation-UNet/dataset/masks'

custom_dataset = CustomDataset(images_dir=images_dir, masks_dir=masks_dir, transform=data_transform)

batch_size = 256
data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)
