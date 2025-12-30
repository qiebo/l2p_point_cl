import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from models.prompt_pool import HyperbolicPromptPool

# Add external path to find pointMLP
# Assumes structure:
# root/
#   point hyperbolic replay/
#   classification_ScanObjectNN/

# Using sys.path to hack import from sibling directory
# Note: 'models' is a package name collision. 
# We need to add classification_ScanObjectNN to path, 
# but simply importing 'models.pointmlp' might still pick up local 'models'.

import sys
import os

target_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../classification_ScanObjectNN'))
if target_path not in sys.path:
    sys.path.append(target_path)

try:
    # Try direct import assuming classification_ScanObjectNN is in path
    from models.pointmlp import pointMLP
except ImportError:
    # Namespace collision likely. 
    # Let's try to import directly from file source as a standalone module
    import importlib.util
    spec = importlib.util.spec_from_file_location("pointmlp_module", os.path.join(target_path, "models/pointmlp.py"))
    pointmlp_module = importlib.util.module_from_spec(spec)
    sys.modules["pointmlp_module"] = pointmlp_module
    spec.loader.exec_module(pointmlp_module)
    pointMLP = pointmlp_module.pointMLP

class PointMLPBackbone(nn.Module):
    def __init__(self, pretrained_path):
        super().__init__()
        # Init PointMLP with 15 classes (ScanObjectNN default) to match weight keys
        self.model = pointMLP(num_classes=15)
        
        # Load Weights
        if os.path.exists(pretrained_path):
            print(f"Loading PointMLP weights from {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            # Check structure of checkpoint
            if 'net' in checkpoint:
                 # The weights are inside 'net' key based on the error message
                 state_dict = checkpoint['net']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Clean keys if necessary (e.g., remove 'module.')
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace("module.", "")
                new_state_dict[name] = v
                
            # Load with strict=False to be safe, but print missing keys
            msg = self.model.load_state_dict(new_state_dict, strict=True)
            print(f"Weight Load Message: {msg}")
        else:
            print(f"WARNING: Weight file not found at {pretrained_path}! Using random init.")

    def forward(self, x):
        # x: (B, N, 3) 
        # pointMLP expects (B, 3, N) generally, but let's check the forward logic again.
        # Original PointMLP forward:
        # xyz = x.permute(0, 2, 1) if input is (B, 3, N) -> (B, N, 3)
        # Wait, pointmlp.py says:
        # def forward(self, x):
        #     xyz = x.permute(0, 2, 1)
        #     x = self.embedding(x)
        # This implies x input should be (B, 3, N).
        
        if x.shape[2] == 3:
            # (B, N, 3) -> (B, 3, N)
            x = x.permute(0, 2, 1)
            
        # Replicate forward pass up to classifier
        # Ref: models/pointmlp.py Model.forward
        
        xyz = x.permute(0, 2, 1) # (B, N, 3)
        x = self.model.embedding(x) # (B, D, N)
        
        for i in range(self.model.stages):
            xyz, x = self.model.local_grouper_list[i](xyz, x.permute(0, 2, 1))
            x = self.model.pre_blocks_list[i](x)
            x = self.model.pos_blocks_list[i](x)
            
        x = F.adaptive_max_pool1d(x, 1).squeeze(dim=-1) # (B, 1024)
        return x

class L2P_PointMLP(nn.Module):
    def __init__(self, num_classes=40, embed_dim=1024, prompt_pool_capacity=20, prompt_length=5, top_k=5,
                 num_tasks=20, prompts_per_task=3, # Added progressive args
                 pretrained_path=r'e:\非全相关\毕业论文\point_l2p\classification_ScanObjectNN\pointMLP-demo1\best_checkpoint.pth'):
        super().__init__()
        
        self.backbone = PointMLPBackbone(pretrained_path)
        
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
            
        self.prompt_pool = HyperbolicPromptPool(
            length=prompt_length,
            embed_dim=embed_dim,
            pool_size=None, # Let pool calculate based on num_tasks * prompts_per_task
            top_k=top_k, 
            prompt_pool=True,
            prompt_key=True,
            num_tasks=num_tasks,
            prompts_per_task=prompts_per_task
        )
        
        self.points_per_prompt = prompt_length
        self.top_k = top_k
        self.embed_dim = embed_dim
        
        self.fused_dim = embed_dim + (top_k * prompt_length * embed_dim)
        
        self.classifier = nn.Linear(self.fused_dim, num_classes)
        
    def forward(self, x, task_id=None):
        with torch.no_grad():
            features = self.backbone(x) # (B, 1024)
        
        features_seq = features.unsqueeze(1)
        
        res = self.prompt_pool(features_seq, task_id=task_id)
        prompted_embedding = res['prompted_embedding']
        
        flattened_embedding = prompted_embedding.reshape(prompted_embedding.shape[0], -1)
        
        logits = self.classifier(flattened_embedding)
        
        return {
            'logits': logits,
            'reduce_sim': res['reduce_sim'],
            'prompt_idx': res['prompt_idx'],
            'embedding': flattened_embedding # Expose for NCM
        }
