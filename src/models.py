import torch
import torch.nn as nn
from torchvision import models
class ImageEncoder(nn.Module):
    def __init__(self, unfreeze_last_block=False):
        super().__init__()

        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Identity()

        # Freeze all layers
        for param in self.cnn.parameters():
            param.requires_grad = False

        # Optionally unfreeze last block
        if unfreeze_last_block:
            for param in self.cnn.layer4.parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.cnn(x)


class MultimodalRegressor(nn.Module):
    def __init__(self, tabular_dim, unfreeze_cnn=False):
        super().__init__()

        self.image_encoder = ImageEncoder(
            unfreeze_last_block=unfreeze_cnn
        )

        self.tabular_mlp = nn.Sequential(
            nn.Linear(tabular_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.regressor = nn.Sequential(
            nn.Linear(512 + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, image, tabular):
        img_feat = self.image_encoder(image)
        tab_feat = self.tabular_mlp(tabular)
        fused = torch.cat([img_feat, tab_feat], dim=1)
        return self.regressor(fused).squeeze(1)
