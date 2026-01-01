import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import models.simple_pointmlp


class SimpleTrainer:
    def __init__(self, args, outdim, device='cuda:0'):
        self.args = args
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        print("Using SimplePointMLP baseline (Frozen backbone + Linear head)...")
        self.model = models.simple_pointmlp.SimplePointMLP(
            num_classes=outdim,
            embed_dim=1024,
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.class_means = {}

    def configure_optimizers(self):
        params_to_optimize = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params_to_optimize.append(param)
        optimizer = torch.optim.Adam(params_to_optimize, lr=self.args.baseline_lr)
        return optimizer

    def train(self, train_loader, val_loader, task_num=0):
        self.model.train()
        optimizer = self.configure_optimizers()

        print(f'<================== Simple Task {task_num} Training ===================>')

        best_acc = 0.0
        epochs = 30

        classes_per_task = self.args.num_classes // self.args.num_task
        seen_classes_limit = (task_num + 1) * classes_per_task

        os.makedirs('checkpoint', exist_ok=True)

        for e in range(epochs):
            tot_loss = 0.0
            tot_size = 0
            start_time = time.time()

            pbar = tqdm(train_loader, desc=f"Epoch {e}")
            for batch in pbar:
                x = batch[0].to(self.device)
                y = batch[1].flatten().long().to(self.device)

                optimizer.zero_grad()
                output = self.model(x)
                logits = output['logits']

                if seen_classes_limit < logits.shape[1]:
                    logits[:, seen_classes_limit:] = -float('inf')

                loss = self.criterion(logits, y)
                loss.backward()
                optimizer.step()

                tot_loss += loss.item() * x.size(0)
                tot_size += x.size(0)
                pbar.set_postfix({'Loss': loss.item()})

            time_cost = time.time() - start_time
            avg_loss = tot_loss / tot_size

            acc = self.validation(val_loader, task_id=task_num)
            print(f'<=== Epoch: {e} | Loss: {avg_loss:.4f} | Val Acc (Linear): {acc:.4f} | Time: {time_cost:.2f}s ===>')

            if acc > best_acc:
                best_acc = acc
                torch.save(self.model.state_dict(), f'checkpoint/Simple_Best_Task{task_num}.pkl')

        return best_acc

    def compute_prototypes(self, train_loader, existing=None):
        print("Computing Prototypes (Simple NCM)...")
        self.model.eval()

        sums = {}
        counts = {}

        with torch.no_grad():
            for batch in tqdm(train_loader, desc="Computing Means"):
                x = batch[0].to(self.device)
                y = batch[1].flatten().long().to(self.device)
                output = self.model(x)
                features = output['embedding']

                for i in range(x.size(0)):
                    cls = y[i].item()
                    feat = features[i]
                    feat = F.normalize(feat, p=2, dim=0)
                    if cls not in sums:
                        sums[cls] = torch.zeros_like(feat)
                        counts[cls] = 0
                    sums[cls] += feat
                    counts[cls] += 1

        prototypes = {}
        for cls in sums:
            mean_vec = sums[cls] / counts[cls]
            mean_vec = F.normalize(mean_vec, p=2, dim=0)
            prototypes[cls] = mean_vec.detach()

        if existing is not None:
            existing.update(prototypes)
            return existing
        return prototypes

    def validation(self, val_loader, task_id=None):
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
                output = self.model(x)
                logits = output['logits']

                if seen_classes_limit < logits.shape[1]:
                    logits[:, seen_classes_limit:] = -float('inf')

                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        return correct / total if total > 0 else 0

    def validation_ncm(self, val_loader):
        print("Validating with Simple NCM...")
        if len(self.class_means) == 0:
            print("No prototypes found! Returning 0.")
            return 0.0

        self.model.eval()
        stored_classes = list(self.class_means.keys())
        proto_tensor = torch.stack([self.class_means[c] for c in stored_classes]).to(self.device)

        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(self.device)
                y = batch[1].flatten().long().to(self.device)
                features = self.model(x)['embedding']
                features = F.normalize(features, p=2, dim=1)
                sims = torch.matmul(features, proto_tensor.T)
                idx = sims.argmax(dim=1)
                labels = torch.tensor([stored_classes[i] for i in idx.cpu().numpy()]).to(self.device)
                correct += (labels == y).sum().item()
                total += y.size(0)

        return correct / total if total > 0 else 0
