import torch
import torch.nn as nn


class FeatureAdapter(nn.Module):
    def __init__(self, embed_dim=1024, bottleneck_dim=128):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.down = nn.Linear(embed_dim, bottleneck_dim)
        self.act = nn.ReLU(inplace=True)
        self.up = nn.Linear(bottleneck_dim, embed_dim)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.down(x)
        x = self.act(x)
        x = self.up(x)
        return residual + x
