import os
import sys

# Setup Environment - MUST BE BEFORE IMPORTING TORCH
# NOTE: CUDA_VISIBLE_DEVICES is now set after parsing args (see main block)
os.environ['PYTORCH_JIT'] = '0' # Disable JIT to fix geoopt gradient issues
os.environ['PYTORCH_NVFUSER_DISABLE'] = 'fallback' # Disable NVFuser fallback

import argparse
import torch
import numpy as np
import datetime
from dataloaders.modelnet import ModelNetDataLoader
from methods.L2P_Trainer import L2P_Trainer

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_task', type=int, default=20)
    parser.add_argument('--dataset', type=str, default='ModelNet40')
    parser.add_argument('--dataroot', type=str, default='../modelnet40_normal_resampled')
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--val_batch_size', type=int, default=32)
    parser.add_argument('--num_classes', type=int, default=40)
    parser.add_argument('--prompts_per_task', type=int, default=3) # Suggestion from user strategy
    parser.add_argument('--top_k', type=int, default=3, help='Number of prompts selected per sample (<= prompts_per_task)')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID to use (e.g., "0", "1", "2", "3")')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of DataLoader workers (0 for Windows, 4-8 for Linux)')
    args = parser.parse_args()
    return args

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self): pass

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

class RemappedSubset(torch.utils.data.Dataset):
    """A wrapper that remaps labels from the underlying dataset."""
    def __init__(self, subset, label_remap):
        self.subset = subset
        self.label_remap = label_remap
    
    def __len__(self):
        return len(self.subset)
    
    def __getitem__(self, idx):
        point_set, label = self.subset[idx]
        old_label = int(label.item() if hasattr(label, 'item') else label[0])
        new_label = self.label_remap.get(old_label, old_label) # Fallback to old if not in remap
        return point_set, np.array([new_label]).astype(np.int32)

def get_modelnet_loader(root, categories, split='train', batch_size=16, label_remap=None, num_workers=4):
    # This helper function filters ModelNetDataLoader for specific categories
    # Since ModelNetDataLoader loads EVERYTHING by default, we need to subset it.
    
    dataset = ModelNetDataLoader(root=root, split=split, normal_channel=False) # L2P usually 3 channels? User said N x 3
    
    # Filter indices
    selected_indices = []
    # dataset.classes is {'airplane': 0, ...}
    # categories is list of keys ['airplane', ...]
    
    target_labels = [dataset.classes[cat] for cat in categories if cat in dataset.classes]
    
    for i in range(len(dataset)):
        # dataset.datapath[i] is (shape_name, path)
        shape_name = dataset.datapath[i][0]
        # Look up label directly from shape name, AVOID loading file
        if shape_name in dataset.classes:
            label = dataset.classes[shape_name]
            if label in target_labels:
                selected_indices.append(i)
            
    subset = torch.utils.data.Subset(dataset, selected_indices)
    
    # Apply Label Remapping if provided
    if label_remap is not None:
        subset = RemappedSubset(subset, label_remap)
    
    # num_workers: Use 0 for Windows, 4-8 for Linux
    loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=(split=='train'), num_workers=num_workers, pin_memory=True)
    return loader

if __name__ == '__main__':
    args = init_args()
    
    # Set GPU AFTER parsing args so user can choose
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print(f"Using GPU: {args.gpu}")
    
    setup_seed(2023)
    
    # Logging
    log_dir = './CIL_logs'
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    sys.stdout = Logger(f'{log_dir}/log_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.txt')
    
    # Class Order (Fixed for reproducibility)
    CATEGORY_ORDER = ['chair', 'sofa', 'airplane', 'bookshelf', 'bed', 'vase', 'monitor', 'table', 'toilet',
                      'bottle', 'mantel', 'tv_stand', 'plant', 'piano', 'car', 'desk', 'dresser',
                      'night_stand', 'glass_box', 'guitar', 'range_hood', 'bench', 'cone', 'tent',
                      'flower_pot', 'laptop', 'keyboard', 'curtain', 'bathtub', 'sink', 'lamp', 'stairs',
                      'door', 'radio', 'xbox', 'stool', 'person', 'wardrobe', 'cup', 'bowl']
    
    # Create a Label Remapping Dict: old_label (alphabetical) -> new_label (CIL order)
    # First, load the original alphabetical order from the dataset
    catfile = os.path.join(args.dataroot, 'modelnet40_shape_names.txt')
    original_cat = [line.rstrip() for line in open(catfile)]
    original_classes = dict(zip(original_cat, range(len(original_cat)))) # {'airplane': 0, ..., 'chair': 8, ...}
    
    # Build remapping: for each category in our CIL order, map old_label -> new_cil_label
    label_remap = {}
    for new_idx, cat_name in enumerate(CATEGORY_ORDER):
        old_idx = original_classes[cat_name]
        label_remap[old_idx] = new_idx
    print(f"Label Remap Created: {label_remap}")
    
    # Init Agent (Trainer)
    # We init with MAX CLASSES (40) to avoid resizing classifier
    agent = L2P_Trainer(args, outdim=args.num_classes)
    
    for n_task in range(args.num_task):
        print(f"\n{'='*20} Task {n_task} {'='*20}")
        
        # Define Categories for this Task
        # Task 0: Cats 0,1. Task 1: Cats 2,3...
        start_idx = n_task * 2
        end_idx = start_idx + 2
        train_cats = CATEGORY_ORDER[start_idx : end_idx]
        
        # Val Categories: All seen so far (0 to current end)
        val_cats = CATEGORY_ORDER[0 : end_idx]
        
        print(f"Training on Categories: {train_cats}")
        print(f"Validating on Categories: {val_cats}")
        
        # Get DataLoaders
        # Check if dataroot exists
        if not os.path.exists(args.dataroot):
             print(f"Warning: Dataset root {args.dataroot} not found!")
             # Create dummy data for testing if real data missing?
             # For now let it crash or warn. Use try/except? 
             # Only wrap loader creation.
             pass

        try:
            train_loader = get_modelnet_loader(args.dataroot, train_cats, split='train', batch_size=args.train_batch_size, label_remap=label_remap, num_workers=args.num_workers)
            val_loader = get_modelnet_loader(args.dataroot, val_cats, split='test', batch_size=args.val_batch_size, label_remap=label_remap, num_workers=args.num_workers)
            
            # Data-Driven Prompt Initialization
            # Calculates task centroid and initializes current task's prompts
            # This helps avoid "Winner-Takes-All" failure by placing new prompts close to new data
            agent.init_prompts_with_centroids(train_loader, task_id=n_task)
            
            # Train
            acc_linear = agent.train(train_loader, val_loader, task_num=n_task)
            print(f"Task {n_task} Training Completed.")
            
            # Compute Prototypes for NCM (Critical for CIL)
            # Pass task_id for correct Prompt Masking during prototype generation
            agent.compute_prototypes(train_loader, task_id=n_task)
            
            # Validate using NCM
            # Pass task_id for correct Prompt Masking during inference
            acc_ncm = agent.validation_ncm(val_loader, task_id=n_task)
            print(f"Task {n_task} Final CIL Accuracy (NCM): {acc_ncm:.4f}")
            print(f"Task {n_task} Linear Head Accuracy (For Reference): {acc_linear:.4f}")
            
        except Exception as e:
            print(f"Error in Task {n_task}: {e}")
            import traceback
            traceback.print_exc()
            break
