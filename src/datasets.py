from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import os

class SatelliteImageDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_files = sorted(os.listdir(image_dir))

        self.transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        return self.transform(image)
import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
import torchvision.transforms as T

class MultimodalDataset(Dataset):
    def __init__(self, csv_path, image_dir, features, target=None,indices = None):
        self.df = pd.read_csv(csv_path)
        self.image_dir = os.path.abspath(image_dir)
        self.features = features
        self.target = target

        self.transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # 🔴 FILTER ONLY SAMPLES WITH EXISTING IMAGES
        self.valid_indices = []

        candidate_indices = indices if indices is not None else range(len(self.df))

        for idx in candidate_indices:

            img_path = os.path.join(self.image_dir, f"{idx}.png")
            if os.path.exists(img_path):
                self.valid_indices.append(idx)

        print(
            f"MultimodalDataset: "
            f"{len(self.valid_indices)} valid samples "
            f"out of {len(self.df)}"
        )

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, i):
        idx = self.valid_indices[i]

        img_path = os.path.join(self.image_dir, f"{idx}.png")
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        tabular = torch.tensor(
            self.df.loc[idx, self.features].values,
            dtype=torch.float32
        )

        if self.target:
            y = torch.tensor(
                self.df.loc[idx, self.target],
                dtype=torch.float32
            )
            return image, tabular, y

        return image, tabular

