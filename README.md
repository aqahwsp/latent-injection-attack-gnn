# latent-injection-attack-gnn
图神经网路中基于特征冲突的潜伏触发式攻击方法研究

## 1. 项目简介

本项目实现了一种面向**图节点分类（Graph Node Classification）**任务的潜伏注入攻击方法。该方法围绕**特征冲突（Feature Conflict）**机理展开，通过将攻击载荷拆分为两类功能不同但相互耦合的注入节点，实现一种“**添加时不触发、删除后生效**”的新型攻击范式。

与传统节点注入攻击不同，本方法并不是在攻击节点被加入图后立即显现攻击效果，而是通过构造：

- **攻击抑制节点（Prelude / Trigger-like Suppressor）**
- **潜伏攻击节点（Latent Malicious Patch Nodes）**

使得两者共同存在时，潜伏攻击节点的恶意作用被抑制；当攻击者后续删除抑制节点后，潜伏攻击节点重新主导模型判别，从而触发误分类攻击。

该设计使攻击具有更强的：

- **隐蔽性**：能够规避主要针对“新增节点 / 新增异常结构”的检测；
- **可操控性**：攻击者可在任意时刻通过“删除抑制器”触发攻击；
- **防御绕过能力**：相较于传统仅关注节点创建或属性修改的防御流程，该攻击更容易在部署后阶段造成破坏。

---

## 2. 方法核心思想

### 2.1 攻击目标

本项目面向图节点分类模型，目标是在**不进行数据投毒**的前提下，对目标节点实施可控误分类攻击，同时尽可能保持模型在干净图上的正常性能。

### 2.2 核心机制

本方法将注入结构拆分为两个功能模块：

#### （1）潜伏攻击模块
潜伏攻击模块本质上是具有攻击能力的注入节点子图，其目标是在后续条件满足时推动目标节点被错误分类。

#### （2）攻击抑制模块
攻击抑制模块在结构与特征上被专门优化，使其在与潜伏攻击模块共同存在时，能够通过特征竞争或梯度主导抑制潜伏模块的即时攻击效果。

### 2.3 触发方式

传统攻击的触发逻辑通常是：

- 加入攻击节点 → 攻击生效

而本项目的触发逻辑为：

- 注入抑制节点 + 潜伏节点 → 攻击暂不生效  
- 删除抑制节点 → 潜伏攻击显现并触发误分类

因此本方法本质上是一种**撤销即触发（Removal-Triggered / Deletion-Triggered）**的潜伏注入攻击框架。

---

## 3. 代码结构

项目当前核心文件如下：

```text
.
├── main_ablation_prelude_updated.py
├── PatchTrainerGNN_DNN_1_updated.py
├── attack_evaluation_updated.py
├── preprocess.py
├── models.py
├── train_clean_gat.py
├── train_clean_gcn.py
└── utils.py
```

### 3.1 `main_ablation_prelude_updated.py`

消融实验主脚本，主要用于研究**攻击抑制节点（prelude / suppressor）**相关机制对攻击效果的影响。

该脚本支持：

- 多数据集实验
- 多 patch 节点规模设置
- trigger ratio（抑制节点比例）扫描
- 不同 backbone（GAT / GCN）对比
- Monte Carlo defense 评估参数设置
- 大规模实验配置管理

如果你要复现实验中“抑制器规模”“触发比例”“数据集差异”这类结果，这个脚本是主要入口。

适合作为**主入口文件**运行完整攻击实验。
---

### 3.2 `PatchTrainerGNN_DNN_1_updated.py`

该文件是本项目最核心的攻击训练模块，实现了**共享式 patch 生成器**与**两阶段攻击结构构造逻辑**，包括：

- 基于 `GATConv + TransformerDecoder` 的 patch 生成器
- 面向目标节点局部子图的 patch 生成
- patch 与 trigger 的组合与冻结策略
- 二值结构与特征的直通估计（Straight-Through Estimator, STE）
- 全 patch / 冻结 patch / trigger 微调机制
- 自动混合精度（AMP）与显存控制策略

其中的关键设计包括：

#### 共享生成器（Shared Generator）
代码使用一个共享的 `TransformerPatchGenerator`，针对不同目标节点的局部子图批量生成 patch，而不是为每个目标单独训练一个注入子图。这使方法更适合批量训练与统一评估。

#### 两阶段训练思想
从代码结构看，训练过程包含明显的阶段化设计：

1. **先学习潜伏攻击 patch**
2. **再在保留第一阶段 patch 的基础上训练 trigger / suppressor 部分**
3. 必要时通过冻结第一阶段 patch 生成器，仅允许第二阶段 trigger 相关结构被更新

这正对应“攻击有效载荷 + 抑制器”协同构造的思想。

#### 二值化机制
项目中实现了针对 patch 邻接矩阵与节点特征的硬二值化与 STE 近似梯度传播，用于构造离散图注入结构。

---

### 3.3 `attack_evaluation_updated.py`

该文件负责统一攻击评估与防御评估，主要功能包括：

- 对指定目标节点集合执行攻击效果统计
- 统一生成 patch / full patch / trigger 组合
- 将 patch 应用到原图中并完成攻击图构造
- 统计攻击前后预测变化
- 执行 Monte Carlo 随机边删除防御评估
- 在局部目标区域中进行随机删边鲁棒性测试
- 汇总攻击成功率、预测分布与防御后表现

该文件说明项目不仅关注“静态攻击是否成功”，还关注：

- 攻击在随机结构扰动下是否仍能保持效果
- 局部防御操作是否会意外触发潜伏攻击
- 防御后预测分布是否出现转移

也就是说，本项目的评估并不只停留在 ASR，而是加入了更贴近真实部署场景的结构随机化分析。

---

### 3.4 `preprocess.py`

数据预处理模块，负责：

- `.npz` 图数据读取
- 稀疏矩阵转 `edge_index`
- 图结构无向化、去自环、coalesce
- 特征归一化
- train / val / test mask 构造
- PyG `Data` 对象构建
- 子图提取与一致性检查

值得注意的是，代码默认采用：

- `train_ratio = 0.7`
- `val_ratio = 0.0`
- `test_ratio = 0.3`

因此如果使用公共数据集或自定义 `.npz` 数据集而没有官方划分，项目会统一重划分。

---

### 3.5 `models.py`

定义项目中使用的图分类模型，包括：

- `GCN`
- `GAT`
- `SmoothGCN`
- `SmoothGAT`

其中：

- `GCN` 与 `GAT` 用于标准干净模型训练和攻击目标模型；
- `SmoothGCN` 与 `SmoothGAT` 引入图结构扰动采样，用于 Monte Carlo 平滑式鲁棒性分析。

这说明项目不仅提供基础分类模型，也兼顾随机平滑防御或鲁棒性统计场景。

---

### 3.6 `train_clean_gat.py` / `train_clean_gcn.py`

两个文件分别用于训练干净的 GAT 与 GCN 模型，主要负责：

- 模型初始化
- 干净图训练
- 验证集精度监控
- 最优模型参数保存
- 测试集精度评估

主脚本会优先尝试调用这两个模块；如果当前划分中没有有效验证集，则会自动退回内置训练逻辑。

---

### 3.7 `utils.py`

通用工具函数，包括：

- 随机种子初始化
- 数据加载辅助函数
- 稀疏矩阵转换
- mask 构造
- 图归一化
- 准确率统计
- Monte Carlo 计数
- 显存清理与内存回收

---

## 4. 方法流程

本项目整体流程可以概括为以下几个阶段。

### 阶段一：训练干净图节点分类模型

首先在原始图数据上训练一个干净分类模型，当前支持：

- GAT
- GCN

该模型作为后续攻击优化时的受害模型。

---

### 阶段二：构造目标节点局部子图表示

对于待攻击目标节点，代码通过 `k_hop_subgraph` 提取其局部邻域子图，并将其作为 patch 生成器的条件输入。

局部子图用于：

- 感知目标节点的局部拓扑结构
- 感知目标节点特征分布
- 指导 patch 与 suppressor 的个性化生成

---

### 阶段三：生成潜伏攻击 patch

项目使用共享式 `TransformerPatchGenerator`，根据目标节点的局部子图生成攻击 patch，包括：

- 注入节点特征
- patch 内部连接结构
- patch 与目标节点之间的连接关系

此阶段主要学习“潜伏攻击有效载荷”。

---

### 阶段四：训练抑制器 / trigger 结构

在保留潜伏 patch 的基础上，继续对 trigger / suppressor 部分进行训练，使其满足：

- 与潜伏 patch 共存时削弱即时攻击效果
- 在结构删除或防御清理后允许潜伏 patch 暴露出攻击能力

从代码实现看，这一阶段通过 patch 冻结、结构拼接、二值图构造和联合前向传播完成。

---

### 阶段五：攻击评估与随机防御评估

最后，框架对攻击效果进行统一评估，包括：

- 干净预测结果
- 注入后预测结果
- full patch / trigger 移除前后预测差异
- 随机删边防御下的表现
- 多次 Monte Carlo 采样结果统计

---

## 5. 项目特点

### 5.1 非投毒节点注入攻击

该方法不依赖于篡改原始训练集标签或样本内容，而是通过**图结构上的注入节点与连接设计**实现攻击。

### 5.2 潜伏式设计

攻击载荷在初始阶段并不直接显现，而是由抑制模块进行掩护，使模型在表面上保持相对正常行为。

### 5.3 删除触发机制

该方法最有代表性的特点是“**删除后触发**”而非“添加后触发”，这是与传统节点注入攻击的重要区别。

### 5.4 可与现有审查流程错位

现实系统往往更关注：

- 新增节点是否异常
- 新增边是否可疑
- 属性更新是否越界

而较少关注“已有节点被删除后，剩余结构是否触发潜伏恶意机制”。本项目正是利用这一点提升攻击隐蔽性。

### 5.5 兼顾大规模实验

代码中加入了：

- batch 版 patch 生成
- 显存保护上限
- AMP 混合精度
- 自动 fallback 训练
- 多数据集统一评估接口

因此较适合系统化实验和消融研究。

---

## 6. 环境要求

建议使用如下环境：

```bash
python >= 3.9
pytorch >= 2.0
torch-geometric >= 2.x
numpy
scipy
tqdm
```

如果你使用 GPU，建议 CUDA 版本与 PyTorch / PyG 兼容。

一个典型安装示例：

```bash
pip install torch torchvision torchaudio
pip install torch-geometric
pip install numpy scipy tqdm
```

> 注意：`torch-geometric` 的安装通常依赖具体的 PyTorch 与 CUDA 版本，建议根据官方安装说明选择匹配版本。

---

## 7. 数据集说明

代码当前支持以下公开图节点分类数据集：

- `Cora`
- `CiteSeer`
- `PubMed`
- `Flickr`

数据加载方式有两类：

### 7.1 公共数据集自动下载
通过 PyG 内置数据集接口自动加载：

- `Planetoid`（Cora / CiteSeer / PubMed）
- `Flickr`

### 7.2 本地 `.npz` 数据
如果当前目录、`data/`、`tmp/` 等路径下存在相应 `.npz` 文件，代码会优先从本地 `.npz` 加载。

---

## 8. 运行方式

### 8.1 运行主实验

```bash
python main_DNN_1_updated.py
```

该脚本用于执行完整主实验流程。

---

### 8.2 运行抑制器相关消融实验

```bash
python main_ablation_prelude_updated.py
```

用于分析：

- patch 节点规模
- trigger / prelude 比例
- 不同数据集表现
- 不同 backbone 表现

---

## 9. 主要参数说明

根据代码实现，实验中较重要的参数包括：

| 参数名 | 含义 |
|---|---|
| `num_patch_nodes` | 注入 patch 的节点数 |
| `num_trigger_nodes` | trigger / 抑制节点数 |
| `trigger_ratio` | trigger 节点数相对 patch 节点数的比例 |
| `target_region_hops` | 目标区域定义的 hop 数 |
| `generator_subgraph_hops` | patch 生成器输入子图的 hop 数 |
| `defense_drop_prob` | 防御评估时随机删边概率 |
| `defense_trials` | Monte Carlo 防御采样次数 |
| `patch_train_epochs` | patch 训练轮数 |
| `trigger_train_epochs` | trigger / prelude 训练轮数 |
| `train_batch_size` | patch 训练批大小 |
| `attack_target_class` | 目标攻击类别 |
| `backbone` | 干净分类模型类型（GAT / GCN） |

---

## 10. 输出结果与评估内容

项目的评估重点通常包括以下几类指标：

- **干净模型准确率**
- **攻击成功率（ASR）**
- **trigger / suppressor 存在时的攻击抑制情况**
- **删除 trigger / suppressor 后的攻击激活情况**
- **随机删边防御下的预测变化**
- **不同数据集与 backbone 下的鲁棒性差异**

从代码结构来看，项目特别强调“**攻击在防御干预后是否反而被激活**”这一安全问题。

---

## 11. 复现建议

为了更稳定地复现实验，建议按如下顺序进行：

1. 先仅训练干净模型，确认数据加载与模型训练正常；
2. 先在单个数据集（如 `Cora`）上测试主实验流程；
3. 再调整 `num_patch_nodes` 与 `trigger_ratio` 观察攻击触发变化；
4. 最后运行 `main_ablation_prelude_updated.py` 做批量消融实验。

如果显存不足，可以优先尝试：

- 减小 `train_batch_size`
- 减少 `num_patch_nodes`
- 关闭或降低大规模评估采样次数
- 使用单一 backbone / 单一数据集先调试

---
