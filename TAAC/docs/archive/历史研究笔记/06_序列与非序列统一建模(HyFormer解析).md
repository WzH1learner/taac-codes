# 序列与非序列特征统一建模：HyFormer 深入解析

> 在探索 TAAC2026 的技术背景时，我们发现 baseline 中的 `PCVRHyFormer` 模型实际上对应了最新的研究成果 **HyFormer (Revisiting the Roles of Sequence Modeling and Feature Interaction in Recommender Systems)**。

## 一、为什么需要"统一建模"？

### 1.1 工业界推荐系统的现状痛点
在传统的大规模推荐系统（CTR预估）中，通常存在两条平行的技术路线：
1. **特征交互（Feature Interaction）**：处理静态属性（如用户画像、物品类别、上下文）。代表模型如 DeepFM、DCN 等。
2. **序列建模（Sequence Modeling）**：处理用户的动态历史行为（如点击序列、购买序列）。代表模型如 DIN、SASRec 等。

### 1.2 现有的分离范式（Separated Paradigm）
工业界最常见的做法是**双塔或晚期融合（Late Fusion）**：
- 序列特征通过 Transformer 或 Pooling 提取为一个定长向量（User Embedding）。
- 非序列特征通过 Embedding 查表得到多个向量。
- 最后将它们拼接在一起，送入顶层的 MLP 进行打分。

**致命缺陷**：
- **特征隔离**：序列中的每一个历史行为，无法在早期与目标候选物品（Target Item）或用户画像发生细粒度的交叉。
- **信息瓶颈**：无论序列多长，最终都被压缩成一个固定维度的向量，导致细粒度的时序交互信息在进入顶层 MLP 前就已经丢失。

---

## 二、HyFormer 的统一建模范式

HyFormer 提出了一种打破序列与非序列边界的**统一架构（Unified Architecture）**。

### 2.1 核心思想：Token 化一切
将所有输入（无论是不是序列）都视为 Token，在一个统一的 Transformer 架构中进行计算。
1. **Sequence Domain**：行为序列天然就是 Token 序列。
2. **Non-Sequence Domain**：将离散特征（如 ID、性别）和连续特征（年龄、历史统计）通过 `NS Tokenizer` 映射为一组 Token。

### 2.2 NS Tokenizer 的设计机制
在 TAAC2026 的 baseline 代码中，`NS Tokenizer` 有两种主要实现：
- **Group 模式**：按语义将非序列特征分组（例如：用户画像组、物品属性组），每组通过 MLP 压缩成 1 个 Token。
- **RankMixer 模式**：将所有非序列特征拼接后，通过线性层直接映射为固定数量的 Tokens（例如 5 个 User Token，2 个 Item Token）。

### 2.3 Query 生成与交叉注意力（Cross-Attention）
如果直接把几百长度的序列 Token 和 几个非序列 Token 拼在一起做 Self-Attention，计算复杂度 $O((L_{seq} + L_{ns})^2)$ 会非常高。

**HyFormer 的解决之道**：
1. **局部序列编码**：先对各个行为域（如点击域、购买域）内部做序列编码（可选用 Transformer 或高效的 LongerEncoder）。
2. **Query 提取**：从每个序列域中提取少量代表性 Token 作为 Query（由 `num_queries` 参数控制，通常为 1 或 2）。
3. **全局交叉融合**：在 `HyFormer Block` 中，这些 Sequence Queries 和 Non-Sequence Tokens 进行互相 Attention。
   - 这实现了序列特征与非序列特征在表征层面的**早期、深度交叉**。

### 2.4 RankMixer 的极致特征交互
经过 HyFormer Block 后，所有的 Token（包含了序列的浓缩信息和属性特征）被送入 RankMixer。
- **Token Mixing**：捕捉不同 Token 之间的相关性（即高阶特征交叉）。
- **Per-Token FFN**：对每个特征子空间进行非线性变换。

---

## 三、统一建模带来的优势与挑战

### 3.1 优势
- **极强的表达能力**：目标物品的特征（NS Token）可以直接去 Attend 用户序列中的每一次历史行为，实现 Target-to-History 的精准匹配。
- **解决冷启动**：当新用户没有序列时，模型可以自动将注意力权重倾斜到用户画像等 NS Token 上。
- **符合 Scaling Law**：统一的 Transformer 架构使得模型可以通过增加深度（Layers）和宽度（d_model）来稳定提升效果，而不会像传统 MLP 那样容易过拟合。

### 3.2 工程挑战
- **计算复杂度增加**：Attention 机制使得计算量陡增，特别是在工业界严格的 RT（响应时间）限制下。
- **内存消耗**：大规模序列需要显存极大的 GPU。
- **解决思路**：这就是为什么 baseline 中提供了 `LongerEncoder`（基于 Top-K 的稀疏注意力）以及 `emb_skip_threshold`（跳过超高基数特征的 embedding 查表）等工程优化手段。

---

## 四、给我们的优化启发

1. **调整 Token 数量**：
   - 尝试增加 `user_ns_tokens` 和 `item_ns_tokens` 的数量，看是否能保留更多的非序列信息。
   - 尝试增加 `num_queries`，让序列特征不至于被过度压缩。
2. **特征交叉增强**：
   - 检查 `RankMixer` 的模式，确保其处于 `full` 模式，以获得最大的特征交叉收益。
3. **序列编码器消融**：
   - 对比 `swiglu`（无注意力）、`transformer`（全注意力）和 `longer`（长序列优化）在实际数据集上的表现差异。

---
**整理时间**：2026-04-25
**技术线索**：基于 TAAC2026 赛题描述与 arXiv 最新论文《HyFormer》联合解析
