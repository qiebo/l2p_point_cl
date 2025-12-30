# 毕业论文项目开发日志 (Thesis Development Log)

**项目名称**: Point Cloud Class-Incremental Learning with L2P (Hyperbolic)
**更新日期**: 2025-12-28

---

## 1. 项目概况 (Project Overview)
本项目旨在解决三维点云数据在**类增量学习 (Class-Incremental Learning, CIL)** 场景下的灾难性遗忘问题。
核心方法采用了 **L2P (Learning to Prompt)** 思想，结合 **双曲几何 (Hyperbolic Geometry)** 进行 Prompt 的查询与匹配，并使用 **PointMLP** 作为强大的特征提取骨干网络。

**核心约束**:
-   **No Replay (无回放)**: 不存储任何旧任务的原始点云数据（Raw Data）。
-   **Frozen Backbone (冻结骨干)**: PointMLP 参数在持续学习过程中保持不变。
-   **Class-Incremental**: 模型在推理时需在所有已见过的类别中进行预测，无任务 ID 提示。

---

## 2. 核心架构 (Technical Architecture)

### 2.1 模型组件
1.  **Backbone**: `PointMLP` (Pre-trained on ScanObjectNN)
    -   *状态*: 冻结 (Frozen)。
    -   *作用*: 提取点云的高层语义特征 (1024-dim)。
    -   *修改*: 修改了 Forward 逻辑，提取分类头之前的 Global Max Pooled 特征。

2.  **L2P Prompt Pool**: `HyperbolicPromptPool`
    -   *状态*: 可训练 (Trainable)。
    -   *机制*: 将输入特征投影到庞加莱球 (Poincare Ball)，计算与 Prompt Keys 之间的双曲距离。
    -   *Prompting*: 选取 Top-K 个最近的 Prompts 拼接到输入特征上。

3.  **Inference Strategy (推理策略)**:
    -   *Training*: 使用线性分类头 (`nn.Linear`) 辅助训练 Prompts，优化 CrossEntropy Loss + Hyperbolic Pull Constraint。
    -   *Testing / Inference*: 采用 **NCM (Nearest Class Mean)** 原型分类器。
        -   原因：线性分类头在 Class-Incremental 且无回放设置下会产生严重的 Logit Bias（倾向于最新任务的类别），导致灾难性遗忘。
        -   实现：每个任务结束后，计算并存储各类的特征中心（Prototypes）。推理时计算样本与所有原型的余弦相似度。

---

## 3. 开发过程与问题解决记录 (Development Journal)

### 3.1 环境与依赖 (Dependencies)
-   **Issue 1: 缺少 `pointnet2_ops`**
    -   *现象*: `ModuleNotFoundError: No module named 'pointnet2_ops'`。这是 PointMLP 原始代码依赖的 CUDA 编译库。
    -   *解决*: 将本地纯 PyTorch 实现的 `models/pointnet_util.py` 复制到 `classification_ScanObjectNN/models/`，并 Patch 了 `models/pointmlp.py` 以调用本地实现，规避了复杂的 CUDA 编译。

-   **Issue 2: 循环引用 (Circular Import)**
    -   *现象*: `ImportError: cannot import name 'HyperbolicPromptPool'`。由于 `models/__init__.py` 贪婪导入导致引用闭环。
    -   *解决*: 清空了 `models/__init__.py` 和 `methods/__init__.py`，切断了自动连锁导入。

### 3.2 训练与硬件 (Training & Backend)
-   **Issue 3: PyTorch JIT / Geoopt 冲突**
    -   *现象*: `RuntimeError: ... DifferentiableGraphBackward returned an invalid gradient`。双曲几何库 `geoopt` 与 PyTorch 的 JIT 编译器在梯度计算上不兼容。
    -   *解决*: 在主程序开头强行设置 `os.environ['PYTORCH_JIT'] = '0'` 禁用 JIT。

-   **Issue 4: Dataloader 导致死机**
    -   *现象*: 程序启动时长时间无响应。原因是在 `get_modelnet_loader` 中遍历读取了 9843 个点云文件来检查标签。
    -   *解决*: 优化过滤逻辑，仅从文件路径/名称中解析标签，避免磁盘 I/O。

### 3.3 核心算法 (Algorithm & Performance)
-   **Issue 5: 严重的灾难性遗忘 (Catastrophic Forgetting)**
    -   *现象*: Task 0 准确率 0.99，但训练完 Task 1 后，总体准确率暴跌至 0.5 左右（且不再上升）。
    -   *分析*: 这是一个典型的 CIL 问题。可训练的线性分类头在 Task 1 只见到了 Class 2、3，CrossEntropy Loss 迫使它抑制 Class 0、1 的响应。由于没有回放，旧类别的权重被遗忘/覆盖。
    -   *解决 (Implemented)*: **引入 NCM (Nearest Class Mean) 策略**。
        -   保持 Prompt 训练逻辑不变。
        -   在每个任务结束后，计算并冻结当前新类别的特征原型 (Prototypes)。
        -   验证时使用 NCM 代替线性分类器，消除了 Logit Bias。

-   **Issue 6: Prompt 排序导致的分类器抖动 (Prompt Permutation Instability)**
    -   *现象*: 代码审查发现 `torch.topk` 返回的 Prompt 索引是按相似度排序的。这意味着对于相似的样本，如果 Prompt 排名稍有变化（如 [P1, P2] 变为 [P2, P1]），拼接后的特征向量顺序会完全颠倒。
    -   *影响*: 线性分类器和 NCM 对输入特征的顺序敏感，这会导致特征空间的剧烈抖动，难以收敛。
    -   *解决*: 在选取 Top-K 后，强制对索引进行排序 (`torch.sort(idx)`)，确保输入特征的排列顺序具有**置换不变性 (Permutation Invariance)**。

-   **Strategy Upgrade: 渐进式 Prompt 策略 (Progressive Prompting)**
    -   *背景*: 为了进一步隔离不同任务的知识，防止 Prompt 相互干扰（Router 漂移）。
    -   *方案*:
        1.  **动态池化**: Prompt Pool 大小设定为 `NumTasks * PromptsPerTask`。
        2.  **任务隔离**: Task $t$ 只能训练属于自己的 $m$ 个 Prompt，旧任务的 Prompt (Key/Value) 被严格冻结。
        3.  **Logit Masking**: 训练时显式 Mask 掉未见类别的 Logits，防止分类器对新类过度拟合（彻底解决 Logit Bias）。
    -   *实施*:
        -   `models/prompt_pool.py`: 实现了基于 `task_id` 的 Masking 机制。
        -   `methods/L2P_Trainer.py`: 引入了 Gradient Hooks 来冻结旧参数，并在 Loss 计算前应用 Logit Mask。

### 3.4 策略优化与最终修正 (Strategy Optimization & Final Fixes)
-   **Issue 7: Validation Accuracy 为 0 (Label Mismatch)**
    -   *现象*: 尽管 Logit Masking 已实施，但验证集准确率始终为 0。
    -   *原因*: `ModelNetDataLoader` 默认按字母顺序分配标签 (0='airplane', 8='chair')，而 CIL 任务流程按 `CATEGORY_ORDER` 定义 (Task 0 包含 'chair', 'sofa')。训练时模型被限制输出 [0, 1]，但 DataLoader 提供的标签是 [8, 35]，导致永远无法匹配。
    -   *解决*: 实现了 **Label Remapping (标签重映射)**。创建了 `label_remap` 字典和 `RemappedSubset` 包装器，将原始字母序标签映射为 CIL 任务序标签 (e.g., 8->0, 35->1)，确保输入与 Logit Mask 对应的范围一致。

-   **Issue 8: 训练崩溃 Loss=inf**
    -   *现象*: 启用 Progressive Masking 后，训练初期的 Loss 为无穷大。
    -   *原因*: `top_k` 设置为 5，但 Task 0 只有 3 个有效 Prompt (`prompts_per_task=3`)。Top-K 强制选择了被 Mask 的未来 Prompt（距离为 `inf`），导致 Pull Constraint Loss 溢出。
    -   *解决*: 将 `top_k` 调整为 **1**。这不仅解决了 Loss 溢出，也符合 "Stable Priority" 的稳定性原则，避免引入过多噪声。

-   **Optimization: 球面中心原型 (Spherical Centroid NCM)**
    -   *改进*: 之前计算 NCM 原型时，直接对特征取平均。为了适配余弦相似度度量（关注方向而非模长），改为 **Pre-Normalization** 策略。
    -   *实现*: `methods/L2P_Trainer.py` 中，在累加样本特征之前，先执行 `F.normalize(feat, p=2)`。最终得到的原型是单位球面上样本点的中心方向。

-   **Optimization: 推理时的时间步约束 (Time-Step Constrained Inference)**
    -   *改进*: 之前的 NCM 推理可能会检索到未来任务的 Prompt（随机初始化状态），引入噪声。
    -   *实现*: 在 `validation_ncm` 中显式传入 `task_id`，利用 Progressive Masking 屏蔽掉所有未来任务的 Prompt，确保推理时只使用已学习到的知识库。

### 3.5 深度稳定化策略 (Deep Stabilization Strategies)
*针对 Task 4 之后 NCM 准确率骤降 (Drop from 90% to 67%) 的问题，实施了以下系统性修复。*

-   **Optimization: 数据驱动的 Prompt 初始化 (Data-Driven Prompt Initialization)**
    -   *问题*: 随机初始化的新 Prompt 距离数据太远，导致新任务样本被经过训练的旧 Prompt "抢走" (Winner-Takes-All)，引发特征错乱。
    -   *解决*: 实现了 `init_prompts_with_centroids`。在每个任务开始前，计算该任务数据的骨干特征中心，并将新 Prompt 的 Key 和 Value 初始化在这个中心附近（加微小噪声以保证多样性）。这为新 Prompt 提供了极强的先验优势。

-   **Optimization: 任务门控 Prompt 选择 (Task-Gated Prompt Selection)**
    -   *问题*: **Prototype 过期 (Prototype Expiry)**。旧类 Prototype 是基于旧 Prompt 计算的。在后续任务推理时，如果旧类样本检索到了新 Prompt（即便不应该），其特征分布会发生偏移，导致与存储的 Prototype 失配。
    -   *解决*: 引入了 **两阶段推理机制**。
        1.  **Task Prediction**: 利用冻结骨干网络的特征（无 Prompt）与存储的 `Task Centroids` 进行余弦相似度匹配，预测样本所属的任务 ID。
        2.  **Gated Extraction**: 强制样本只使用预测任务的 Prompts 进行特征提取。
    -   *效果*: 彻底阻断了"用新 Prompt 提取旧类特征"的路径，保证了特征的一致性和 Prototype 的有效性。

-   **Optimization: Key 完全冻结 (Key Freezing)**
    -   *问题*: Router 漂移。如果允许更新 Key，新任务的训练可能会改变 Query-Key 的匹配关系，导致旧类样本在未来检索错误的 Prompt。
    -   *解决*: 修改 Gradient Hook，将所有 Key 的梯度强制置零。结合 Centroid Initialization，Keys 在初始化后即固定，仅训练 Prompt Values。这提供了最强的稳定性保障。

---

## 4. 后续规划 (Future Work)
-   [ ] **验证 NCM 效果**: 确认 Task 1 及其后续任务的 NCM 准确率是否稳定。
-   [ ] **超参微调**: 调整 Pull Constraint 的权重 $\lambda$ 或 Prompt Pool 大小。
-   [ ] **论文图表**: 导出 `CIL_logs` 目录下的日志数据，绘制 Accuracy Curve。

---

## 5. 详细实验配置 (Experiment Configuration)
*本节记录项目初始运行时的关键参数设置，便于复现。*

### 5.1 环境设置 (Environment)
*   **OS**: Windows
*   **CUDA**: Single GPU (ID: 0)
*   **PyTorch JIT**: `Disabled` (Env Var: `PYTORCH_JIT='0'`, `PYTORCH_NVFUSER_DISABLE='fallback'`) —— 用于修复 Geoopt 库的梯度 bug。

### 5.2 数据集 (Dataset)
*   **Dataset**: ModelNet40 (Normal Resampled)
*   **Path**: `e:\非全相关\毕业论文\modelnet40_normal_resampled` (Relative: `../modelnet40_normal_resampled`)
*   **Config**:
    *   `num_task`: 20 (Total 40 classes, 2 classes per task)
    *   `batch_size`: 16 (Train) / 32 (Val)
    *   `num_workers`: 0 (禁用多进程以避免 Windows 死锁)

### 5.3 模型参数 (Model Hyperparameters)
*   **Backbone**: `PointMLP` (Weights: `pointMLP-demo1/best_checkpoint.pth`)
    *   *Frozen*: Yes
    *   *Input Points*: 1024
    *   *Feature Dim*: 1024
*   **Prompt Pool (L2P)**:
    *   `embed_dim`: 1024
    *   `pool_size`: 20 (Total prompts available)
    *   `prompt_length`: 5 (Number of learnable tokens per prompt)
    *   `top_k`: 5 (Selected prompts per sample)
    *   `selection_metric`: Hyperbolic Distance (Poincare Ball)

### 5.4 训练设置 (Training Settings)
*   **Optimizer**: Adam
    *   `lr`: 0.01 (Optimizing only Prompt Pool & Linear Head parameters)
*   **Loss Function**:
    *   `CrossEntropyLoss` (Classification) + `0.1 * ReduceSim` (Hyperbolic Pull Constraint)
*   **Epochs**: 30 per task
*   **Evaluation**:
    *   *Method*: Nearest Class Mean (NCM) on Global Prototypes.
    *   *Validation Set*: Cumulative (Includes all seen classes 0 to current task).
