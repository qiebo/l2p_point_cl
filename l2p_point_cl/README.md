# Project CLI Reference

This project exposes a single training entry point:

```bash
python "point hyperbolic replay/PointNet_CL_CIL.py" [args...]
```

## Method Options

| Method | Description |
| --- | --- |
| `simple_baseline` | Frozen PointMLP backbone + linear head training + NCM evaluation. |
| `lae_adapter_ncm` | Online/Offline adapter (EMA) + ensemble NCM diagnostics. |
| `l2p` | Prompt pool with L2P-style selection and NCM evaluation. |
| `coda_prompt` | L2P with per-task orthogonality regularization (CODA-Prompt). |

## CLI Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `--num_task` | `20` | Number of tasks in class-incremental training. |
| `--dataset` | `ModelNet40` | Dataset name (logging only). |
| `--dataroot` | `../modelnet40_normal_resampled` | Path to ModelNet40 data root. |
| `--train_batch_size` | `16` | Training batch size. |
| `--val_batch_size` | `32` | Validation batch size. |
| `--num_classes` | `40` | Total number of classes. |
| `--prompts_per_task` | `3` | Prompt pool size per task (prompt-based methods). |
| `--top_k` | `3` | Top-k prompts selected per sample (prompt-based methods). |
| `--selection_metric` | `hyperbolic` | Prompt selection metric: `hyperbolic` or `cosine`. |
| `--map_scale` | `0.1` | Scale for expmap0 into Poincaré ball (hyperbolic selection). |
| `--prompt_key_lr` | `0.001` | Learning rate for prompt keys (prompt-based methods). |
| `--method` | `l2p` | Training method: `l2p`, `coda_prompt`, `lae_adapter_ncm`, `simple_baseline`. |
| `--adapter_dim` | `128` | Adapter bottleneck dimension (LAE). |
| `--ema_decay` | `0.99` | EMA decay for offline adapter (LAE). |
| `--ensemble_alpha` | `0.7` | Offline weight in ensemble NCM (LAE). |
| `--online_lr` | `0.01` | Learning rate for online adapter/head (LAE). |
| `--distill_lambda` | `0.1` | Distillation weight for online↔offline features (LAE). |
| `--orth_lambda` | `0.1` | Orthogonality loss weight (CODA-Prompt). |
| `--baseline_lr` | `0.001` | Learning rate for simple baseline head. |
| `--pretrained_path` | `e:\\非全相关\\毕业论文\\point_l2p\\classification_ScanObjectNN\\pointMLP-demo1\\best_checkpoint.pth` | Path to PointMLP pretrained weights. |
| `--gpu` | `0` | GPU id to use. |
| `--num_workers` | `4` | DataLoader workers (0 recommended on Windows). |

## Example Commands

**Simple baseline**
```bash
python "point hyperbolic replay/PointNet_CL_CIL.py" \
  --method simple_baseline \
  --baseline_lr 1e-3
```

**CODA-Prompt**
```bash
python "point hyperbolic replay/PointNet_CL_CIL.py" \
  --method coda_prompt \
  --selection_metric cosine \
  --orth_lambda 0.1
```

**LAE adapter**
```bash
python "point hyperbolic replay/PointNet_CL_CIL.py" \
  --method lae_adapter_ncm \
  --online_lr 1e-3 \
  --distill_lambda 0.1
```
