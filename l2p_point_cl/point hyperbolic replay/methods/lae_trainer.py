import copy
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import models.lae_pointmlp


class LAE_Trainer:
    def __init__(self, args, outdim, device='cuda:0'):
        self.args = args
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        print("Using LAE_PointMLP with Feature Adapter...")
        self.online_model = models.lae_pointmlp.LAE_PointMLP(
            num_classes=outdim,
            embed_dim=1024,
            adapter_dim=args.adapter_dim,
        ).to(self.device)
        self.offline_model = copy.deepcopy(self.online_model).to(self.device)
        self.offline_model.eval()
        for param in self.offline_model.parameters():
            param.requires_grad = False

        self.criterion = nn.CrossEntropyLoss()
        self.class_means_online = {}
        self.class_means_offline = {}

    def configure_optimizers(self):
        params_to_optimize = []
        for name, param in self.online_model.named_parameters():
            if param.requires_grad:
                params_to_optimize.append(param)
        optimizer = torch.optim.Adam(params_to_optimize, lr=0.01)
        return optimizer

    def train(self, train_loader, val_loader, task_num=0):
        self.online_model.train()
        optimizer = self.configure_optimizers()

        print(f'<================== LAE Task {task_num} Training ===================>')

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
                output = self.online_model(x)
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
                torch.save(self.online_model.state_dict(), f'checkpoint/LAE_Best_Task{task_num}.pkl')

        return best_acc

    def update_offline_adapter(self, momentum):
        online_adapter = self.online_model.adapter
        offline_adapter = self.offline_model.adapter
        for p_off, p_on in zip(offline_adapter.parameters(), online_adapter.parameters()):
            p_off.data = momentum * p_off.data + (1.0 - momentum) * p_on.data

    def compute_prototypes(self, model, train_loader):
        model.eval()
        sums = {}
        counts = {}
        with torch.no_grad():
            for batch in tqdm(train_loader, desc="Computing Means"):
                x = batch[0].to(self.device)
                y = batch[1].flatten().long().to(self.device)
                output = model(x)
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
        return prototypes

    def validation(self, val_loader, task_id=None):
        self.online_model.eval()
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
                output = self.online_model(x)
                logits = output['logits']

                if seen_classes_limit < logits.shape[1]:
                    logits[:, seen_classes_limit:] = -float('inf')

                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        return correct / total if total > 0 else 0

    def validation_ncm(self, val_loader, alpha=0.7):
        print("Validating with LAE Ensemble NCM...")
        if len(self.class_means_online) == 0 or len(self.class_means_offline) == 0:
            print("No prototypes found! Returning 0.")
            return 0.0

        self.online_model.eval()
        self.offline_model.eval()

        stored_classes = list(self.class_means_online.keys())
        proto_online = torch.stack([self.class_means_online[c] for c in stored_classes]).to(self.device)
        proto_offline = torch.stack([self.class_means_offline[c] for c in stored_classes]).to(self.device)

        correct_online = 0
        correct_offline = 0
        correct_ensemble = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(self.device)
                y = batch[1].flatten().long().to(self.device)

                feat_on = self.online_model(x)['embedding']
                feat_off = self.offline_model(x)['embedding']

                feat_on = F.normalize(feat_on, p=2, dim=1)
                feat_off = F.normalize(feat_off, p=2, dim=1)

                logits_on = torch.matmul(feat_on, proto_online.T)
                logits_off = torch.matmul(feat_off, proto_offline.T)
                logits = alpha * logits_off + (1.0 - alpha) * logits_on

                idx_on = logits_on.argmax(dim=1)
                labels_on = torch.tensor([stored_classes[i] for i in idx_on.cpu().numpy()]).to(self.device)
                correct_online += (labels_on == y).sum().item()

                idx_off = logits_off.argmax(dim=1)
                labels_off = torch.tensor([stored_classes[i] for i in idx_off.cpu().numpy()]).to(self.device)
                correct_offline += (labels_off == y).sum().item()

                idx = logits.argmax(dim=1)
                labels = torch.tensor([stored_classes[i] for i in idx.cpu().numpy()]).to(self.device)
                correct_ensemble += (labels == y).sum().item()
                total += y.size(0)

        acc_online = correct_online / total if total > 0 else 0
        acc_offline = correct_offline / total if total > 0 else 0
        acc_ensemble = correct_ensemble / total if total > 0 else 0

        print("DIAGNOSTICS:")
        print(f"  [Online-only]  NCM Accuracy: {acc_online:.4f}")
        print(f"  [Offline-only] NCM Accuracy: {acc_offline:.4f}")
        print(f"  [Ensemble]     NCM Accuracy: {acc_ensemble:.4f}")

        return acc_ensemble
