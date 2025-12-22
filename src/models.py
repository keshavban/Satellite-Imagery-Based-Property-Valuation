import torch
import torch.nn as nn
from torchvision import models

class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Identity()  # remove classifier

    def forward(self, x):
        return self.cnn(x)
class MultimodalRegressor(nn.Module):
    def __init__(self, tabular_dim):
        super().__init__()

        # Image branch
        self.image_encoder = ImageEncoder()

        # Tabular branch
        self.tabular_mlp = nn.Sequential(
            nn.Linear(tabular_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # Fusion head
        self.regressor = nn.Sequential(
            nn.Linear(512 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, image, tabular):
        img_feat = self.image_encoder(image)
        tab_feat = self.tabular_mlp(tabular)

        fused = torch.cat([img_feat, tab_feat], dim=1)
        return self.regressor(fused).squeeze(1)
