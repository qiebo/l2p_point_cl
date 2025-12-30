import torch
import torch.nn as nn
from tqdm import tqdm
import time
import models.l2p_pointmlp

import torch.nn.functional as F

class L2P_Trainer:
    def __init__(self, args, outdim, device='cuda:0'):
        self.args = args
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Use PointMLP by default as per user request
        print("Using L2P_PointMLP with Pre-trained Weights...")
        self.model = models.l2p_pointmlp.L2P_PointMLP(
            num_classes=outdim, 
            embed_dim=1024,
            prompt_pool_capacity=20, # Ignored now
            prompt_length=5, 
            top_k=1, # Fixed: Must be <= prompts_per_task (3). User recommended k=1 for stability.
            # If k > prompts_per_task, we select masked (inf) prompts -> Loss=inf.
            # User said "stable priority k=1, not recommend k>=3".
            # BUT user instruction 2.2 says "keep your existing injection... but ensure...".
            # Plan said "Reduce default top_k to 1... or keep at prompts_per_task".
            # prompts_per_task is 3. Let's set top_k = prompts_per_task (3).
            # No, user 2.1 says "recommends k=1". But let's stick to prompts_per_task for now or 1?
            # Let's use args.top_k if available or hardcode to 3 (prompts_per_task).
            # Wait, prompts_per_task default is 3. Let's use that.
            num_tasks=args.num_task,
            prompts_per_task=args.prompts_per_task 
        ).to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        
        # NCM Prototypes: Dict mapping class_index (int) -> mean_vector (Tensor)
        self.class_means = {}
        
        # Task Centroids: Dict mapping task_id (int) -> centroid (Tensor)
        # Used for Task-Gated Prompt Selection to prevent prototype expiry
        self.task_centroids = {} 
        
    def init_prompts_with_centroids(self, train_loader, task_id):
        """
        Data-Driven Initialization:
        Initialize the prompts (keys and values) for the current task using the 
        Centroid of the training data features.
        This provides a "warm start" and ensures new prompts are closer to new data
        than old frozen prompts.
        """
        print(f"Initializing Prompts for Task {task_id} using Data Centroids...")
        self.model.eval()
        
        # 1. Compute Centroid
        all_feats = []
        with torch.no_grad():
            for batch in tqdm(train_loader, desc="Computing Task Centroid"):
                x = batch[0].to(self.device)
                # Ensure we use backbone features only, no prompting yet
                features = self.model.backbone(x) # (B, D)
                features = F.normalize(features, p=2, dim=1) # Normalize consistent with NCM
                all_feats.append(features)
        
        all_feats = torch.cat(all_feats, dim=0)
        centroid = torch.mean(all_feats, dim=0) # (D,)
        centroid = F.normalize(centroid, p=2, dim=0)
        
        # 2. Assign to Prompts
        # Target indices for current task
        prompts_per_task = self.model.prompt_pool.prompts_per_task
        start_idx = task_id * prompts_per_task
        end_idx = (task_id + 1) * prompts_per_task
        
        print(f" Assigning Centroid to Prompts [{start_idx}:{end_idx}]")
        
        # Modify Parameters in-place
        with torch.no_grad():
            # Init Keys: Use centroid direction with unique noise per prompt
            if self.model.prompt_pool.prompt_key is not None:
                # Scale centroid down to be safe inside Poincare Ball (norm 0.1)
                # Add unique noise to each key to ensure diversity
                for i in range(start_idx, end_idx):
                    key_noise = torch.randn_like(centroid) * 0.01
                    self.model.prompt_pool.prompt_key.data[i] = centroid * 0.1 + key_noise
            
            # Init Values: Use centroid direction with unique noise per prompt
            # Prompt shape: (Pool, Length, Dim)
            for i in range(start_idx, end_idx):
                target_val = centroid.view(1, 1, -1).repeat(1, self.model.prompt_pool.length, 1)
                # Generate unique noise for each prompt to avoid identical values
                noise = torch.randn_like(target_val) * 0.01
                self.model.prompt_pool.prompt.data[i] = target_val + noise

    def configure_optimizers(self):
        # Only optimize Prompt Pool and Classifier
        # Backbone is already frozen in L2P_PointNet __init__
        
        params_to_optimize = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Note: We rely on gradient hooks for fine-grained prompt freezing
                params_to_optimize.append(param)
        
        # Use Adam as standard
        optimizer = torch.optim.Adam(params_to_optimize, lr=0.01) # Tuning might be needed
        return optimizer

    def train(self, train_loader, val_loader, val_original=None, task_num=0, ref_model=None):
        self.model.to(self.device)
        self.model.train() # This sets training mode (Backbone is eval due to frozen)
        optimizer = self.configure_optimizers()
        
        print(f'<================== L2P Task {task_num} Training ===================>')
        
        # --- PROGRESSIVE FREEZING HOOKS ---
        # Freeze all prompts EXCEPT those belonging to current task
        prompts_per_task = self.model.prompt_pool.prompts_per_task
        start_idx = task_num * prompts_per_task
        end_idx = (task_num + 1) * prompts_per_task
        
        def freeze_grad_hook_values(grad):
            """Freeze old and future task prompt values"""
            g = grad.clone()
            # Zero out grads for old tasks
            if start_idx > 0:
                g[:start_idx] = 0
            # Zero out grads for future tasks (though masked in fwd, safe to enforce)
            if end_idx < g.shape[0]:
                g[end_idx:] = 0
            return g
        
        def freeze_grad_hook_keys(grad):
            """Freeze ALL prompt keys to prevent router drift (Strategy 3)"""
            # Return zero gradient for all keys - they're initialized well via centroids
            return torch.zeros_like(grad)
            
        # Register hooks
        # Note: prompt and prompt_key are Parameters in HyperbolicPromptPool
        h1 = self.model.prompt_pool.prompt.register_hook(freeze_grad_hook_values)
        h2 = None
        if self.model.prompt_pool.prompt_key is not None:
             h2 = self.model.prompt_pool.prompt_key.register_hook(freeze_grad_hook_keys)
        # ----------------------------------
        
        best_acc = 0.0
        epochs = 30 # Default from old code
        
        classes_per_task = self.args.num_classes // self.args.num_task
        seen_classes_limit = (task_num + 1) * classes_per_task
        
        for e in range(epochs):
            tot_loss = 0.0
            tot_size = 0
            start_time = time.time()
            
            pbar = tqdm(train_loader, desc=f"Epoch {e}")
            for batch in pbar:
                # Batch format from ModelNetDataLoader: (point_set, label)
                
                x = batch[0].to(self.device)
                y = batch[1].flatten().long().to(self.device)
                
                optimizer.zero_grad()
                
                # Pass task_id for masking future prompts
                output = self.model(x, task_id=task_num)
                logits = output['logits'] # (B, NumClasses)
                
                # --- MASKED CROSS ENTROPY ---
                # Mask unseen classes (indices >= seen_classes_limit) with -inf
                # This ensures they have 0 probability in Softmax and no gradient
                if seen_classes_limit < logits.shape[1]:
                    logits[:, seen_classes_limit:] = -float('inf')
                # ----------------------------
                
                # Loss Calculation
                ce_loss = self.criterion(logits, y)
                pull_constraint = output['reduce_sim'] # Hyperbolic Distance
                
                # Total Loss = CE + lambda * Pull
                # Lambda typically 0.1
                loss = ce_loss + 0.1 * pull_constraint
                
                loss.backward()
                optimizer.step()
                
                tot_loss += loss.item() * x.size(0)
                tot_size += x.size(0)
                
                pbar.set_postfix({'Loss': loss.item(), 'CE': ce_loss.item(), 'Pull': pull_constraint.item()})
                
            time_cost = time.time() - start_time
            avg_loss = tot_loss / tot_size
            
            # Validation (Linear Head) - For monitoring only
            # Pass task_num to ensure correct masking of unseen logit outputs
            acc = self.validation(val_loader, task_id=task_num)
            print(f'<=== Epoch: {e} | Loss: {avg_loss:.4f} | Val Acc (Linear): {acc:.4f} | Time: {time_cost:.2f}s ===>')
            
            if acc > best_acc:
                best_acc = acc
                torch.save(self.model.state_dict(), f'checkpoint/L2P_Best_Task{task_num}.pkl')
        
        # Remove hooks to avoid compounding
        h1.remove()
        if h2 is not None:
            h2.remove()
                
        return best_acc

    def compute_prototypes(self, train_loader, task_id=None):
        print("Computing Prototypes (NCM) for current task...")
        self.model.eval()
        
        # Temporary storage for current pass
        sums = {}
        counts = {}
        
        # Storage for task centroid (backbone-only features)
        task_feats = []
        
        with torch.no_grad():
            for batch in tqdm(train_loader, desc="Computing Means"):
                x = batch[0].to(self.device)
                y = batch[1].flatten().long().to(self.device)
                
                # Compute Task Centroid (backbone-only, no prompts)
                backbone_feat = self.model.backbone(x)  # (B, D)
                backbone_feat_norm = F.normalize(backbone_feat, p=2, dim=1)
                task_feats.append(backbone_feat_norm)
                
                # Use task_id for correct prompt masking during prototype generation
                output = self.model(x, task_id=task_id)
                # We need features BEFORE classifier
                features = output['embedding'] # (B, D)
                
                for i in range(x.size(0)):
                    cls = y[i].item()
                    feat = features[i]
                    
                    # USER STRATEGY UPGRADE: L2 Normalize BEFORE aggregation
                    # This computes the "Spherical Centroid" rather than "Euclidean Centroid"
                    feat = F.normalize(feat, p=2, dim=0)
                    
                    if cls not in sums:
                        sums[cls] = torch.zeros_like(feat)
                        counts[cls] = 0
                    
                    sums[cls] += feat
                    counts[cls] += 1
        
        # Calculate Task Centroid and store
        if task_id is not None:
            all_task_feats = torch.cat(task_feats, dim=0)
            task_centroid = torch.mean(all_task_feats, dim=0)
            task_centroid = F.normalize(task_centroid, p=2, dim=0)
            self.task_centroids[task_id] = task_centroid.detach()
            print(f" Task {task_id} Centroid saved.")
        
        # Calculate Means and Update global storage
        for cls in sums:
            mean_vec = sums[cls] / counts[cls]
            mean_vec = F.normalize(mean_vec, p=2, dim=0) # Normalize again after averaging
            self.class_means[cls] = mean_vec.detach() # Explicitly detach/keep in memory
            
        print(f"Prototypes updated. Total classes stored: {len(self.class_means)}")

    def validation(self, val_loader, task_id=None):
        # Original Linear Validation
        # NOTE: Using task_id to mask logits for unseen classes
        self.model.eval()
        correct = 0
        total = 0
        
        if task_id is not None:
            classes_per_task = self.args.num_classes // self.args.num_task
            seen_classes_limit = (task_id + 1) * classes_per_task
        else:
            seen_classes_limit = self.args.num_classes

        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(self.device)
                y = batch[1].flatten().long().to(self.device)
                
                # Pass task_id for Consistent Prompting
                output = self.model(x, task_id=task_id)
                logits = output['logits']
                
                # Apply Masking to Linear Head Validation as well
                if seen_classes_limit < logits.shape[1]:
                    logits[:, seen_classes_limit:] = -float('inf')
                
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        
        return correct / total if total > 0 else 0

    def validation_ncm(self, val_loader, task_id=None):
        # Task-Gated NCM Inference with DIAGNOSTICS
        print("Validating with Task-Gated NCM (Diagnostic Mode)...")
        if len(self.class_means) == 0:
            print("No prototypes found! Returning 0.")
            return 0.0
            
        self.model.eval()
        
        # Metrics
        correct_pred = 0 # Using predicted task
        correct_oracle = 0 # Using true task (Oracle)
        correct_task = 0 # Task Prediction Accuracy
        total = 0
        
        stored_classes = list(self.class_means.keys())
        proto_tensor = torch.stack([self.class_means[c] for c in stored_classes]).to(self.device)
        
        use_task_gating = len(self.task_centroids) > 0
        if use_task_gating:
            task_ids = sorted(self.task_centroids.keys())
            task_centroid_tensor = torch.stack([self.task_centroids[t] for t in task_ids]).to(self.device)
            print(f" Using Task-Gated Inference with {len(task_ids)} task centroids.")
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(self.device)
                y = batch[1].flatten().long().to(self.device)
                
                # Derive ground-truth task_id from labels for Oracle
                # Assuming categories are ordered sequentially by task
                classes_per_task = self.args.num_classes // self.args.num_task
                true_task_ids = y // classes_per_task # (B,)
                
                if use_task_gating:
                    # --- Step 1: Predict Tasks ---
                    backbone_feat = self.model.backbone(x)  # (B, D)
                    backbone_feat_norm = F.normalize(backbone_feat, p=2, dim=1)
                    task_sim = torch.matmul(backbone_feat_norm, task_centroid_tensor.T)
                    predicted_tasks_indices = task_sim.argmax(dim=1)
                    predicted_tasks = torch.tensor([task_ids[i] for i in predicted_tasks_indices.cpu().numpy()]).to(self.device)
                    
                    # Task Prediction Accuracy
                    correct_task += (predicted_tasks == true_task_ids).sum().item()
                    
                    # --- Step 2: Extract Features & NCM ---
                    feature_list_pred = []
                    feature_list_oracle = []
                    
                    for i in range(x.size(0)):
                        sample = x[i:i+1]
                        
                        # A) Predicted Task Prompt
                        pred_t = predicted_tasks[i].item()
                        out_pred = self.model(sample, task_id=pred_t)
                        feature_list_pred.append(out_pred['embedding'])
                        
                        # B) Oracle Task Prompt
                        true_t = true_task_ids[i].item()
                        # If true task is NOT yet in stored centroids (e.g. valid on future classes?), clamp?
                        # But validation set should only contain seen tasks usually.
                        # For robustness, fallback to pred if true_t not in known tasks (rare case)
                        oracle_t = true_t
                        out_oracle = self.model(sample, task_id=oracle_t)
                        feature_list_oracle.append(out_oracle['embedding'])
                        
                    features_pred = torch.cat(feature_list_pred, dim=0)
                    features_oracle = torch.cat(feature_list_oracle, dim=0)
                    
                else:
                    # Fallback (No gating yet)
                    output = self.model(x, task_id=task_id)
                    features_pred = output['embedding']
                    features_oracle = features_pred
                    
                # --- Classification ---
                # A) Predicted
                norm_feat_pred = F.normalize(features_pred, p=2, dim=1)
                sims_pred = torch.matmul(norm_feat_pred, proto_tensor.T)
                idx_pred = sims_pred.argmax(dim=1)
                labels_pred = torch.tensor([stored_classes[i] for i in idx_pred.cpu().numpy()]).to(self.device)
                correct_pred += (labels_pred == y).sum().item()
                
                # B) Oracle
                norm_feat_oracle = F.normalize(features_oracle, p=2, dim=1)
                sims_oracle = torch.matmul(norm_feat_oracle, proto_tensor.T)
                idx_oracle = sims_oracle.argmax(dim=1)
                labels_oracle = torch.tensor([stored_classes[i] for i in idx_oracle.cpu().numpy()]).to(self.device)
                correct_oracle += (labels_oracle == y).sum().item()
                
                total += y.size(0)
        
        acc_pred = correct_pred / total if total > 0 else 0
        acc_oracle = correct_oracle / total if total > 0 else 0
        acc_task = correct_task / total if total > 0 else 0
        
        print(f"DIAGNOSTICS:")
        print(f"  [Predicted-Gated] NCM Accuracy: {acc_pred:.4f}")
        print(f"  [Oracle-Gated]    NCM Accuracy: {acc_oracle:.4f}")
        print(f"  [Task Prediction] Accuracy:     {acc_task:.4f}")
        
        return acc_pred
