from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torch
import pandas as pd
import os

imageSize = 32


class AntennaDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0]+'.png')
        image = Image.open(img_path).convert("RGB")
        label = torch.tensor(self.img_labels.iloc[idx, 1:].tolist(), dtype=float)
        label = label
        title = self.img_labels.iloc[idx, 0]

        if self.transform:
            image = self.transform(image)

        return image.float(), label.float(), title

transform = transforms.Compose([
                             transforms.Grayscale(num_output_channels=1),
                             transforms.Resize(imageSize),
                             transforms.ToTensor(),
                             transforms.Normalize((0.5,), (0.5,)),
                         ])