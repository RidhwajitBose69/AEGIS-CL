"""
IDS-CL — Model definition.

A small MLP for tabular network-traffic features (NSL-KDD: 41 → encoded ~122 features).
Explicitly split into .backbone and .head so we can:
  - freeze backbone for NCM / linear-probe diagnosis
  - apply EWC / severity-weighted EWC selectively
  - swap head for new task classes
"""

import random
import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class MLPBackbone(nn.Module):
    """Feature extractor for tabular data."""
    def __init__(self, input_dim: int, hidden1: int = 256, hidden2: int = 128, feat_dim: int = 64):
        super().__init__()
        self.feat_dim = feat_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden2, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ContinualIDS(nn.Module):
    """
    Full model = backbone (feature extractor) + head (classifier).
    Same API as the CNN version so strategies.py works unchanged.
    """
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.backbone = MLPBackbone(input_dim=input_dim)
        self.head = nn.Linear(self.backbone.feat_dim, num_classes)

    def forward(self, x, return_features=False):
        feats = self.backbone(x)
        logits = self.head(feats)
        if return_features:
            return logits, feats
        return logits

    def extract_features(self, x):
        with torch.no_grad():
            return self.backbone(x)


if __name__ == "__main__":
    set_seed(42)
    m = ContinualIDS(input_dim=122, num_classes=5)
    x = torch.randn(8, 122)
    logits, feats = m(x, return_features=True)
    print(f"logits: {logits.shape}, feats: {feats.shape}")
    n_params = sum(p.numel() for p in m.parameters())
    print(f"Total params: {n_params:,}")
