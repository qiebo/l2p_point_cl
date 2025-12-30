import torch
import torch.nn as nn
from models.prompt_pool import HyperbolicPromptPool

class PointNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        # Simplified PointNet Feature Extractor (up to Global Feature 1024)
        self.feature_layer = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
    
    def forward(self, x):
        # x: (B, N, 3) expected input format to align with old dataset, 
        # but PointNet conv1d expects (B, 3, N).
        # We perform permute here.
        if x.shape[2] == 3:
            x = x.permute(0, 2, 1)
            
        x = self.feature_layer(x)
        x, _ = torch.max(x, 2) # (B, 1024)
        return x

class L2P_PointNet(nn.Module):
    def __init__(self, num_classes=10, embed_dim=1024, prompt_pool_capacity=20, prompt_length=5, top_k=5):
        super().__init__()
        self.backbone = PointNetBackbone()
        
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval() # Force eval mode
            
        self.prompt_pool = HyperbolicPromptPool(
            length=prompt_length,
            embed_dim=embed_dim,
            pool_size=prompt_pool_capacity,
            top_k=top_k, 
            prompt_pool=True,
            prompt_key=True # Learnable keys
        )
        
        # Fusion Strategy: Concat
        # Input: Global Feature (1) + Prompts (top_k)
        # Total Tokens = 1 + (top_k * prompt_length)
        # Flattened Dim = Total Tokens * Embed Dim
        
        self.points_per_prompt = prompt_length # assuming pool returns [B, top_k * length, C]
        self.top_k = top_k
        self.embed_dim = embed_dim
        
        self.fused_dim = embed_dim + (top_k * prompt_length * embed_dim)
        
        # Classifier
        self.classifier = nn.Linear(self.fused_dim, num_classes)
        
    def forward(self, x):
        # x: (B, N, 3)
        with torch.no_grad():
            features = self.backbone(x) # (B, 1024)
        
        # Reshape for Prompt Pool (expecting sequence-like input for compatibility)
        # (B, 1, 1024)
        features_seq = features.unsqueeze(1)
        
        res = self.prompt_pool(features_seq)
        
        # res['prompted_embedding'] is (B, Total_Len, 1024)
        prompted_embedding = res['prompted_embedding']
        
        # Flatten
        flattened_embedding = prompted_embedding.reshape(prompted_embedding.shape[0], -1)
        
        logits = self.classifier(flattened_embedding)
        
        return {
            'logits': logits,
            'reduce_sim': res['reduce_sim'], # This is actually distance in our hyperbolic impl (minimize it)
            'prompt_idx': res['prompt_idx']
        }
