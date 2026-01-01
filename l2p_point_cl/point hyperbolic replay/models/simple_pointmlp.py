import torch
import torch.nn as nn

from models.l2p_pointmlp import PointMLPBackbone


class SimplePointMLP(nn.Module):
    def __init__(self, num_classes=40, embed_dim=1024,
                 pretrained_path=r'e:\非全相关\毕业论文\point_l2p\classification_ScanObjectNN\pointMLP-demo1\best_checkpoint.pth'):
        super().__init__()
        self.backbone = PointMLPBackbone(pretrained_path)
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)
        logits = self.classifier(features)
        return {
            'logits': logits,
            'embedding': features
        }
