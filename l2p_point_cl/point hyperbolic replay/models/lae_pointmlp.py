import torch
import torch.nn as nn

from models.adapter import FeatureAdapter
from models.l2p_pointmlp import PointMLPBackbone


class LAE_PointMLP(nn.Module):
    def __init__(self, num_classes=40, embed_dim=1024, adapter_dim=128,
                 pretrained_path=r'e:\非全相关\毕业论文\point_l2p\classification_ScanObjectNN\pointMLP-demo1\best_checkpoint.pth'):
        super().__init__()
        self.backbone = PointMLPBackbone(pretrained_path)
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        self.adapter = FeatureAdapter(embed_dim=embed_dim, bottleneck_dim=adapter_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)
        adapted = self.adapter(features)
        logits = self.classifier(adapted)
        return {
            'logits': logits,
            'embedding': adapted
        }
