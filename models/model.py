# model.py
"""
Advanced image classifier for binary pothole detection.

Features:
- Pretrained ResNet50 backbone (configurable)
- Squeeze-and-Excitation (SE) attention block
- Strong classifier head with dropout and optional MC Dropout
- Utilities to convert model.state_dict() <-> list[numpy.ndarray] (useful for Flower)
- Save / load helpers

Usage:
    from model import get_model, model_to_ndarrays, ndarrays_to_model, save_model, load_model
    model = get_model(backbone="resnet50", pretrained=True, dropout=0.5)

Notes:
- If you train with BCEWithLogitsLoss prefer the model without final Sigmoid.
- Downloading pretrained weights requires internet on first run.
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        b, c, _, _ = x.size()
        se = torch.mean(x.view(b, c, -1), dim=2)  # global avg pool -> (B, C)
        se = F.relu(self.fc1(se))
        se = torch.sigmoid(self.fc2(se))
        se = se.view(b, c, 1, 1)
        return x * se


class SpatialAttention(nn.Module):
    """Simple spatial attention (conv on pooled channels)."""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # channel-wise pooling
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        cat = torch.cat([max_pool, avg_pool], dim=1)
        attn = self.sigmoid(self.conv(cat))
        return x * attn


class AdvancedResNetClassifier(nn.Module):
    """
    Advanced classifier using a ResNet backbone + SE + spatial attention.
    Output: scalar in [0,1] (Sigmoid).
    """
    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        dropout: float = 0.5,
        use_se: bool = True,
        use_spatial_attn: bool = True,
        mc_dropout: bool = False,
    ):
        super().__init__()
        backbone = backbone.lower()
        self.use_se = use_se
        self.use_spatial_attn = use_spatial_attn
        self.mc_dropout = mc_dropout

        # Load backbone
        if backbone == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            num_feats = self.backbone.fc.in_features  # 2048
        elif backbone == "resnet34":
            self.backbone = models.resnet34(pretrained=pretrained)
            num_feats = self.backbone.fc.in_features  # 512
        elif backbone == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained)
            num_feats = self.backbone.fc.in_features  # 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Remove the original fc and avgpool to use our own pooling + attention
        # Keep everything up to layer4 inclusive
        self.backbone_fc_replaced = True
        # Cut off final fc and avgpool
        self.backbone.fc = nn.Identity()
        # ResNet has avgpool already; we'll call backbone(x) -> feature map if necessary.
        # To get the 2D feature map, we forward until layer4 using the original modules:
        # but easiest is to use self.backbone and manually apply avgpool if needed.

        # Attention blocks applied to final conv feature map (optional)
        if self.use_se:
            self.se = SEBlock(num_feats, reduction=16)
        else:
            self.se = nn.Identity()

        if self.use_spatial_attn:
            self.spatial_attn = SpatialAttention(kernel_size=7)
        else:
            self.spatial_attn = nn.Identity()

        # Pooling and classifier head
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        # Optionally MC Dropout: dropout layers active during eval for uncertainty estimation
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_feats, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(p=dropout),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Input: x (B, 3, H, W), expected normalized as ImageNet (mean/std).
        Output: (B, 1) probability.
        """
        # Use resnet stem + layers to get final conv feature map
        # Emulate forward until last conv feature map (before avgpool)
        # Reference: torchvision resnet forward: conv1->bn1->relu->maxpool->layer1->layer2->layer3->layer4->avgpool->fc
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)  # final conv feature map, shape (B, C, Hf, Wf)

        # Attention on feature map
        if self.use_se:
            x = self.se(x)
        if self.use_spatial_attn:
            x = self.spatial_attn(x)

        # Global pooling + classifier
        x = self.global_pool(x)  # (B, C, 1, 1)
        x = self.classifier(x)  # (B, 1) after flatten + final sigmoid
        return x

    def enable_mc_dropout(self):
        """Enable dropout at eval time (for MC Dropout)."""
        self.mc_dropout = True
        self.train()  # ensure dropout layers are in train mode while still using eval for BN if needed

    def disable_mc_dropout(self):
        """Disable MC dropout and use normal eval mode."""
        self.mc_dropout = False
        self.eval()


# -----------------------
# Helper functions
# -----------------------
def get_model(
    backbone: str = "resnet50",
    pretrained: bool = True,
    dropout: float = 0.5,
    use_se: bool = True,
    use_spatial_attn: bool = True,
    mc_dropout: bool = False,
) -> nn.Module:
    """
    Create the model.

    Returns:
        model (nn.Module)
    """
    model = AdvancedResNetClassifier(
        backbone=backbone,
        pretrained=pretrained,
        dropout=dropout,
        use_se=use_se,
        use_spatial_attn=use_spatial_attn,
        mc_dropout=mc_dropout,
    )
    return model


def model_to_ndarrays(model: nn.Module) -> List[np.ndarray]:
    """
    Convert model.state_dict() to list of numpy arrays for Flower (or any aggregator).
    Order matches state_dict().values().
    """
    return [val.cpu().numpy() for val in model.state_dict().values()]


def ndarrays_to_model(model: nn.Module, arrays: List[np.ndarray]) -> None:
    """
    Load list of numpy arrays into model.state_dict() (in-place).
    arrays must be in same order as model.state_dict().values().
    """
    keys = list(model.state_dict().keys())
    if len(keys) != len(arrays):
        raise ValueError(f"Length mismatch: {len(keys)} keys vs {len(arrays)} arrays")
    state_dict = {}
    for k, arr in zip(keys, arrays):
        state_dict[k] = torch.tensor(arr)
    model.load_state_dict(state_dict, strict=True)


def save_model(model: nn.Module, path: str) -> None:
    """
    Save PyTorch state_dict to path.
    """
    torch.save(model.state_dict(), path)


def load_model(model: nn.Module, path: str, map_location: Optional[str] = None) -> None:
    """
    Load state_dict into model.
    """
    map_loc = None if map_location is None else map_location
    sd = torch.load(path, map_location=map_loc)
    model.load_state_dict(sd)
