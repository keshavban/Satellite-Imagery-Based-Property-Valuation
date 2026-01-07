import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image

class MultimodalDataset(Dataset):
    def __init__(self, df, image_dir, tabular_features, transform):
        self.image_dir = image_dir
        self.tabular_features = tabular_features
        self.transform = transform

        valid_rows = []
        valid_image_ids = []

        for idx in df.index:
            img_path = os.path.join(image_dir, f"{idx}.png")
            if os.path.exists(img_path):
                valid_rows.append(df.loc[idx])
                valid_image_ids.append(idx)

        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)
        self.image_ids = valid_image_ids

        print(f"Loaded {len(self.df)} samples with images")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.image_dir, f"{img_id}.png")

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        tabular = torch.tensor(
            self.df.loc[idx, self.tabular_features]
            .astype("float32")
            .values,
            dtype=torch.float32

        )

        y = torch.tensor(
            self.df.loc[idx, "log_price"],
            dtype=torch.float32
        )

        return image, tabular, y
