import torch
import torch.nn as nn
from methods.L2P_Trainer import L2P_Trainer
import argparse
import sys
import os

# Add path for pointMLP just in case verify_model is run locally
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../classification_ScanObjectNN')))

def test_model_dims():
    print("Testing L2P_PointMLP Integration...")
    
    # Init Args
    args = argparse.Namespace()
    
    # Init Trainer (Will load L2P_PointMLP)
    try:
        trainer = L2P_Trainer(args, outdim=40, device='cpu')
    except Exception as e:
        print(f"FAILED to initialize trainer: {e}")
        import traceback
        traceback.print_exc()
        return

    model = trainer.model
    
    # Create Dummy Data
    B, N, C = 4, 1024, 3
    x = torch.randn(B, N, C)
    y = torch.randint(0, 5, (B,)).long()
    
    # Forward
    print("Running Forward Pass...")
    try:
        out = model(x)
        logits = out['logits']
        reduce_sim = out['reduce_sim']
        prompt_idx = out['prompt_idx']
        
        print(f"Logits Shape: {logits.shape} (Expected: [{B}, 40])")
        print(f"Reduce Sim: {reduce_sim.item()}")
        print(f"Prompt Indices: {prompt_idx.shape}")
        
    except Exception as e:
        print(f"FAILED during forward pass: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Backward
    print("Running Backward Pass...")
    ce_loss = trainer.criterion(logits, y)
    loss = ce_loss + 0.1 * reduce_sim
    loss.backward()
    
    print("Backward Pass Successful!")
    print("Test Passed.")

if __name__ == "__main__":
    test_model_dims()
