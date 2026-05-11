# 外部 TAAC2026 代码仓库参考笔记

> 目的：记录公开仓库中与当前 PCVRHyFormer baseline 相关的思路，用于后续实验设计、数据分析和工程优化。当前笔记基于仓库 README / pyproject 可见内容，不等价于完整代码审查。

## 1. 当前本地 baseline 观察

### 训练日志确认

新任务日志显示当前训练配置已经正确生效：

| 项 | 当前值 | 说明 |
|---|---:|---|
| `num_epochs` | 10 | 最大训练轮数 |
| `patience` | 3 | AUC 连续 3 次不提升则早停 |
| `num_queries` | 2 | 每个序列域生成 2 个 query token |
| `user_ns_tokens` | 5 | user NS token 数 |
| `item_ns_tokens` | 2 | item NS token 数 |
| `emb_skip_threshold` | 1000000 | 超高基数特征不建 embedding |
| `reinit_sparse_after_epoch` | 2 | 第 2 轮后开始 sparse reinit |
| `reinit_cardinality_threshold` | 100000 | vocab > 100000 的 embedding 才重置 |

模型结构日志：

```text
PCVRHyFormer model created: num_ns=8, T=16, d_model=64, rank_mixer_mode=full
Total parameters: 239,931,649
Sparse params: 98 tensors, 237,435,776 parameters (Adagrad lr=0.05)
Dense params: 380 tensors, 2,495,873 parameters (AdamW lr=0.0001)
```

关键理解：

- **Sparse Embedding 是参数主体**：约 237M / 240M，优化重点不能只看 Transformer/MLP。
- **`T=16` 满足 `64 % 16 == 0`**：当前 rank mixer full mode 结构合法。
- **`emb_skip_threshold=1000000` 已生效**：日志显示 `seq_b skipped 1/13`、`seq_c skipped 3/11`，说明若 vocab 超过 100 万，会用零向量替代 embedding，节省显存。
- **后续需要关注 reinit 日志**：理想情况应出现 `Re-initialized X high-cardinality Embeddings (vocab>100000), kept Y`，其中 `kept` 应明显大于旧配置的 `1`。

## 2. 仓库一：Puiching-Memory/TAAC_2026

地址：https://github.com/Puiching-Memory/TAAC_2026

### 项目定位

该仓库更像一个规范化的实验工作区，而不是单个 notebook/baseline：

- 使用 `src/taac2026` 包结构。
- 提供 CLI：`taac-train`、`taac-evaluate`、`taac-search`、`taac-package-train`、`taac-dataset-eda` 等。
- 使用 `uv` 管理环境，并区分 CPU / CUDA profile。
- 引入 `optuna` 做搜索，`pytest/hypothesis` 做测试，`rich/loguru` 做工程化输出。

### 数据理解

README 对字段结构的描述很清楚：

| 类别 | 字段理解 |
|---|---|
| ID/Label | `user_id`, `item_id`, `label_type`, `label_time`, `timestamp` |
| User dense | `user_dense_feats_{61,87}` 是用户 embedding 特征；`user_dense_feats_{62-66,89-91}` 与对应 user int 数组对齐 |
| Item int | 大部分是标量，`item_int_feats_11` 是数组 |
| 序列域 | `domain_a_seq_{38-46}`、`domain_b_seq_{67-79,88}`、`domain_c_seq_{27-37,47}`、`domain_d_seq_{17-26}` |

对我们的启发：

- 当前 PCVRHyFormer 把 user/item dense 投影成 token，但还没有充分利用 **dense 与 int array 对齐** 这一点。
- 可以专门做一次 EDA：统计每个 list 特征长度分布、空值比例、vocab size、与 label 的关系。
- `user_dense_feats_{62-66,89-91}` 可能不是普通 dense，而是与对应离散 id 绑定的数值侧信息，后续可考虑在 tokenizer 内按元素融合。

### 工程启发

该仓库的工程化方向值得参考：

- 用配置文件定义 experiment，而不是分散在 `run.sh`、`train.py`、`infer.py`。
- 打包训练包用于平台上传，减少漏文件和跨文件参数不一致。
- 提供 EDA CLI 和搜索 CLI，适合后续系统化调参。

短期可借鉴点：

1. 建一个小型参数快照机制：训练启动时把最终 `args` 写入 ckpt 与 log。
2. 增加 EDA 脚本，输出 vocab/缺失率/序列长度/label 分布。
3. 增加配置一致性检查，避免 `train.py` 与 `eval/infer.py` fallback 漂移。

## 3. 仓库二：hojiahao/TAAC2026

地址：https://github.com/hojiahao/TAAC2026

### 项目定位

该仓库提出了一个更激进的 UniRec 架构设想：把序列建模和特征交互统一到一个同构可堆叠 backbone 中。

README 提到融合方向：

| 来源 | 思想 |
|---|---|
| OneTrans | 统一分词，S-token 共享参数、NS-token 独立参数 |
| InterFormer | backbone 前显式 feature cross |
| HSTU 2.0 | SiLU gated attention、semi-local attention、attention truncation、MoT |
| DIN | target-aware interest extraction |
| Kimi | block attention residuals |
| 本项目 | hybrid attention mask |

### 模型设计启发

该仓库给出的分阶段结构：

1. **统一分词**：每个非序列字段成 token，序列每步 concat 后投影成 token。
2. **骨干前预处理**：NS token self-attention、target-aware interest、target fusion、MoT。
3. **统一序列组装**：NS token + 多域序列 token + target/interest/mot token。
4. **统一 backbone**：hybrid attention mask，让 NS 全双向、序列局部/半局部、target 全可见。
5. **预测头**：target token 输出接 MLP。

对当前 PCVRHyFormer 的启发：

- 当前模型已经有 query generator + HyFormer block，但 target item 的显式查询能力还可以增强。
- 可以考虑加入轻量版 DIN：用 item 表征作为 query，对四个行为序列做 target-aware pooling。
- 可以考虑 NS-token 之间先做一层 self-attention 或 feature cross，再进入 HyFormer。
- 对长序列可尝试更激进的截断/半局部 attention，而不是固定 `seq_c/seq_d=512` 全量 Transformer。

### 损失函数启发

该仓库提出 `CombinedAUCLoss = WeightedBCE + PairwiseBPR`。

对当前 baseline 的启发：

- 现在只有 BCE / Focal。AUC 是排序指标，BCE 不直接优化排序。
- 后续可做小实验：
  - BCE baseline
  - Focal loss
  - BCE + sampled pairwise ranking loss
- 注意 pairwise loss 会增加 batch 内正负样本构造成本，先做轻量版本即可。

### 工程优化启发

README 中列出的优化方向：

| 方向 | 可借鉴点 |
|---|---|
| BF16 混合精度 | 当前训练看起来还未启用 AMP，可优先尝试 BF16 autocast |
| Gradient Checkpointing | 如果加深模型或增大序列长度，可以节省激活显存 |
| torch.compile | 可能提升推理/训练速度，但平台兼容性需验证 |
| Warmup + Cosine LR | 当前 AdamW lr 固定，可尝试 warmup/cosine |
| persistent_workers + prefetch | 当前 DataLoader 可进一步优化 worker 生命周期 |
| GPU 端累积评估 | 当前 evaluate 最后拼 CPU tensor，可考虑减少同步 |
| Attention truncation | 深层只处理最近 token，降低长序列成本 |

## 4. 与当前 baseline 的差距地图

| 维度 | 当前 PCVRHyFormer | 外部仓库启发 | 优先级 |
|---|---|---|---|
| 配置管理 | `run.sh` + argparse + infer fallback | experiment config / packaging | 高 |
| 数据分析 | 依赖 schema，缺独立 EDA | EDA CLI、字段结构分析 | 高 |
| NS tokenization | RankMixer 压缩为 5+2 token | per-field token / feature cross | 中 |
| 序列建模 | 4 域 Transformer，固定截断 | target-aware pooling、semi-local、truncation | 中 |
| Loss | BCE/Focal | WeightedBCE + PairwiseBPR | 中 |
| Optimizer | Embedding Adagrad + Dense AdamW | 混合优化器、warmup/cosine | 中 |
| 显存/速度 | 无 AMP/compile/checkpoint | BF16、compile、checkpoint | 中 |
| 平台可靠性 | best_model sidecar 曾出问题 | package-train/输出检查 | 高 |

## 5. 建议的下一批实验

### 实验 A：确认当前修正版 baseline

目标：验证配置修复是否带来更稳定的训练/评估链路。

关注日志：

- `Epoch N Validation | AUC, LogLoss`
- `earlyStopping counter`
- `Re-initialized X high-cardinality Embeddings (vocab>100000), kept Y`
- best_model 目录大小是否大于 0，是否包含 `model.pt`, `train_config.json`, `schema.json`

### 实验 B：Embedding reinit 消融

| 实验 | 参数 |
|---|---|
| 当前 | `reinit_after=2`, `threshold=100000` |
| 关闭 reinit | `reinit_after=999`, `threshold=999999999` |
| 更激进 MEDA | `reinit_after=1`, `threshold=0` 或低阈值 |

目的：确认 one-epoch / MEDA 机制在该数据上的真实收益。

### 实验 C：序列长度消融

| 实验 | `seq_max_lens` |
|---|---|
| 当前 | `seq_a:256,seq_b:256,seq_c:512,seq_d:512` |
| 省算力 | `seq_a:128,seq_b:128,seq_c:256,seq_d:256` |
| 更长行为 | `seq_a:256,seq_b:256,seq_c:768,seq_d:768`（若显存允许） |

目的：判断长序列是否贡献主要 AUC，还是只增加训练时间。

### 实验 D：Loss 消融

| 实验 | 参数 |
|---|---|
| BCE | `--loss_type bce` |
| Focal | `--loss_type focal --focal_alpha 0.1 --focal_gamma 2.0` |
| Pairwise 混合 | 需新增 sampled pairwise loss |

## 6. 数据分析待办

建议后续新增 `research/code/eda_plan.md` 或脚本，至少输出：

1. `label_type` 正负样本比例。
2. 每个 int/list 特征的 vocab size、缺失率、top value 占比。
3. 每个序列域长度分布：P50/P90/P99/max。
4. 序列最近行为时间间隔分布。
5. `user_id`、`item_id` 重复率和长尾程度。
6. 被 `emb_skip_threshold` 跳过的字段列表。
7. `reinit_cardinality_threshold=100000` 下会被重置的字段列表。

## 7. 结论

短期不要急着大改模型。当前最重要的是：

1. 跑通修正版 baseline，并拿到首个有效平台 eval 分数。
2. 用 EDA 找到 vocab、序列长度、正负样本比例、跳过/重置字段的真实分布。
3. 做 reinit、seq length、loss 三组低成本消融。
4. 再考虑引入 target-aware pooling、NS feature cross、BF16/warmup/cosine 等结构或训练优化。

## 8. 实验 A/B 复盘与单实验推荐（2026-04-27）

### 实验 A 结果

实验 A 已经完成训练和平台 eval：

| 指标 | 结果 |
|---|---:|
| 最佳验证 AUC | 0.8571 |
| 最佳 epoch | 1 |
| 后续验证 AUC | 0.8289 / 0.8541 / 0.8509 |
| 平台 eval AUC | 0.797713 |
| 实际训练 | 4 epoch 后 early stopping |

关键现象：

- 验证集在第 1 个 epoch 达到最佳，后续没有超过第 1 轮。
- 验证 AUC 从 0.8571 到 0.8289 的跌幅较大，说明训练存在明显震荡或泛化退化。
- 平台 eval AUC 与验证 AUC 存在较大 gap，说明下一步目标不只是提高 valid AUC，而是降低过拟合并提升测试泛化。

### 实验 B 结果

实验 B 只调整：

```bash
--sparse_lr 0.01 --dropout_rate 0.1 --patience 5
```

其他结构和 selective reinit 策略保持不变。

| 指标 | 结果 |
|---|---:|
| 平台 eval AUC | 约 0.791 |
| epoch1 valid AUC / LogLoss | 0.854510 / 0.228862 |
| epoch2 valid AUC / LogLoss | 0.826170 / 0.259480 |
| epoch3 valid AUC / LogLoss | 0.852282 / 0.232249 |
| epoch4 valid AUC / LogLoss | 0.850219 / 0.236205 |
| epoch5 valid AUC / LogLoss | 0.846593 / 0.235312 |

关键现象：

- 平台 eval 从实验 A 的约 `0.797713` 降到约 `0.791`。
- epoch1 valid AUC 也从实验 A 的 `0.8571` 降到 `0.8545`，说明更低 `sparse_lr` 与更高 `dropout_rate` 可能削弱了有效 embedding 信号。
- epoch2 仍然暴跌到 `0.8262`，说明问题不是简单由 sparse lr 太大或 dropout 太小造成。
- 每次 selective reinit 后日志均显示 `restored optimizer state for 0 low-cardinality params`，这比普通超参数问题更可疑。

### 对之前代码改动的判断

之前提交中同步 `train.py`、`run.sh`、`eval/infer.py` 的模型配置是必要且合理的：

- `num_queries=2`、`user_ns_tokens=5`、`item_ns_tokens=2` 使 `T=16`，满足 `d_model=64` 下 `64 % 16 == 0`。
- `emb_skip_threshold=1000000` 避免超高基数序列特征直接创建巨大 embedding，节省显存并降低极端 ID 记忆风险。
- `reinit_cardinality_threshold=100000` 修复了旧默认 `threshold=0` 导致几乎所有 embedding 被重置的问题。
- `trainer.py` 中重建 Adagrad 后只恢复未重置 embedding 的 optimizer state，符合 selective reinit 的设计：被重置的高基数 embedding 冷启动，低基数 embedding 保留 Adagrad 历史；但实验 B 暴露出 state restore 可能没有真正生效，需要优先修复或确认。
- checkpoint sidecar 可靠性修复有助于 eval 端优先从 `train_config.json` 和 `schema.json` 重建模型，减少 fallback 配置不一致风险。

这些改动主要解决的是“配置正确性”和“训练机制不明显错误”的问题。实验 A/B 的结果说明，下一步不应继续盲目调 `sparse_lr` / `dropout_rate`，而应优先验证 selective reinit 与 Adagrad state restore 是否按预期工作。

### 如果一次只能开一个实验

如果已经修复 `trainer.py` 的 Adagrad state restore 逻辑，推荐回到实验 A 原参数：

```bash
--sparse_lr 0.05 --dropout_rate 0.01 --patience 3
```

保持其他关键结构不变：

| 参数 | 保持值 |
|---|---:|
| `num_queries` | 2 |
| `user_ns_tokens` | 5 |
| `item_ns_tokens` | 2 |
| `d_model` | 64 |
| `num_hyformer_blocks` | 2 |
| `num_heads` | 4 |
| `emb_skip_threshold` | 1000000 |
| `reinit_sparse_after_epoch` | 2 |
| `reinit_cardinality_threshold` | 100000 |
| `loss_type` | bce |

选择这组的理由：

1. **最干净验证**：只修复 optimizer state restore，不同时引入新的超参数变化。
2. **保留 selective reinit 假设**：继续让高基数 embedding 冷启动，同时让低基数 embedding 保留 Adagrad 历史。
3. **直接验证关键日志**：`old_state candidates` 和 `restored optimizer state` 应明显大于 0，而不是继续为 0。
4. **对照实验 A**：结构、数据、参数回到已知 baseline，便于判断修复是否改善 valid/eval。

预期判断标准：

| 现象 | 结论 |
|---|---|
| `restored` 仍为 0 | state restore 仍未生效，继续查 optimizer state key 或参数匹配 |
| `restored` 明显大于 0，epoch2 不再大幅跌破 epoch1 | reinit 后丢失低基数 Adagrad 历史是关键问题之一 |
| `restored` 明显大于 0，但 valid/eval 仍差 | selective reinit 本身可能不适合当前数据，需要做 reinit 消融 |
| valid 接近或超过 0.8571，且平台 eval 高于 0.797713 | 修复方向有效，可再考虑小幅调参 |

如果暂时不能改代码，只能通过平台 `script_args` 做一次参数实验，则推荐关闭 reinit 做消融：

```bash
--reinit_sparse_after_epoch 999 --reinit_cardinality_threshold 999999999 --sparse_lr 0.05 --dropout_rate 0.01 --patience 5
```

## 2026-04-29 实验进展更新

### Full reinit Transformer OOM-safe

为解决云平台第一步 forward OOM，将配置调整为：

```bash
--batch_size 128
--seq_max_lens seq_a:128,seq_b:128,seq_c:256,seq_d:256
--amp
--amp_dtype bfloat16
--reinit_sparse_after_epoch 0
--reinit_cardinality_threshold 0
```

该配置官方 eval AUC 已达到 `0.804`，高于旧实验 A 的 `0.797713`。这说明 full reinit + OOM-safe 配置不是单纯降配，至少在官方 eval 上带来了正收益。

补充 valid 曲线：

| Epoch | Valid AUC | Valid LogLoss |
|---|---:|---:|
| 1 | 0.855688 | 0.228127 |
| 2 | 0.857923 | 0.230407 |
| 3 | 0.858478 | 0.226766 |
| 4 | 0.860081 | 0.227152 |
| 5 | 0.860202 | 0.224951 |
| 7 | 0.860406 | 0.225311 |

Best valid 在 epoch7，而不是 epoch1。结合官方 eval `0.804`，当前阶段应从 one-epoch 思维切换到“full reinit 下训练 6-8 epoch，提交 best checkpoint 做官方 eval”。

### Full reinit SwiGLU OOM-safe

第二张卡将唯一结构变量改为：

```bash
--seq_encoder_type swiglu
```

其他关键配置保持：

```bash
--batch_size 128
--seq_max_lens seq_a:128,seq_b:128,seq_c:256,seq_d:256
--reinit_sparse_after_epoch 0
--reinit_cardinality_threshold 0
--sparse_lr 0.05
--dropout_rate 0.01
--amp
--amp_dtype bfloat16
```

当前日志显示：

| Epoch | Valid AUC | Valid LogLoss |
|---|---:|---:|
| 1 | 0.856409 | 0.228545 |
| 2 | 0.858486 | 0.228177 |
| 3 | 0.859721 | 0.225764 |
| 4 | 0.860341 | 0.225996 |
| 5 | 0.859695 | 0.225844 |
| 6 | 0.861175 | 0.225692 |
| 7 | 0.860802 | 0.227167 |
| 8 | 0.859335 | 0.226809 |

关键判断：

- `swiglu + full reinit` 已经超过历史 one-epoch valid 最优 `0.8571`，说明 one-epoch 现象在该配置下被明显缓解。
- full reinit 日志为 `Re-initialized 97 high-cardinality Embeddings (vocab>0), kept 1`，参数确实生效。
- 该实验的官方 eval 是当前最重要下一步；如果 eval 高于 `0.804`，后续应围绕轻量序列编码器继续调参。
- 不建议再复刻 `reinit_after=2, threshold=100000` 作为主实验，因为旧实验已经证明其信息增益有限；除非要在同一 OOM-safe 配置下做严格对照。

### Adagrad reinit 内存实现建议

同事版本通过 clone 全部 optimizer state 再重建 optimizer，能避免部分引用问题，但在 full reinit 下会临时保留大量即将丢弃的 Adagrad 累积张量。更省显存的实现是：

1. 先执行 reinit，得到 `reinit_ptrs`。
2. 只保存 `ptr not in reinit_ptrs` 的旧 optimizer state。
3. 删除旧 optimizer 并 `torch.cuda.empty_cache()`。
4. 重建 Adagrad，只恢复未重置参数的 state。

这样 full reinit 时只保留少量 kept 参数的历史状态，避免 epoch 末尾因 optimizer state clone 造成额外显存压力。

## 2026-04-29 结构分析更新：不要只调参

当前最强官方结果：

| 实验 | 官方 eval AUC |
|---|---:|
| Transformer full reinit OOM-safe | 0.80332 / 0.804 左右 |
| SwiGLU full reinit OOM-safe | 0.806956 |

这说明 `seq_encoder_type=swiglu` 方向有效，但只靠 `sparse_lr/dropout` 小调不一定足够。下一步应围绕数据/schema 暴露出的结构问题做高信息增益实验。

### RankMixer NS token 当前的隐性问题

当前线上实验使用：

```bash
--ns_groups_json ""
--ns_tokenizer_type rankmixer
--user_ns_tokens 5
--item_ns_tokens 2
```

由于 `--ns_groups_json ""`，训练不会读取 `train/ns_groups.json`，而是把 46 个 user int 特征和 14 个 item int 特征分别当作 singleton groups。`RankMixerNSTokenizer` 的真实逻辑是：

1. 按 group 顺序把每个 fid 的 embedding 均值向量拼成长向量。
2. 将长向量平均切成 `user_ns_tokens` / `item_ns_tokens` 个 chunk。
3. 每个 chunk 线性投影到一个 NS token。

用 sample parquet 的 fid 顺序估算：

```text
user: 46 fids, 5 tokens, emb_dim=64
total_emb_dim=2944, chunk_dim=589, pad=1
WARNING: 589 % 64 != 0，token 边界会切进 feature embedding 中间

item: 14 fids, 2 tokens, emb_dim=64
total_emb_dim=896, chunk_dim=448
448 % 64 == 0，刚好每个 token 7 个 item fids
```

这意味着当前 user NS token 并不完全对应“若干完整特征的组合”，而会出现 fid 53/63/91/100 等 embedding 被跨 token 切开。它可能仍然有效，但不够语义干净。

### 推荐结构实验

优先实验：

```bash
--seq_encoder_type swiglu
--ns_tokenizer_type group
--ns_groups_json "${SCRIPT_DIR}/ns_groups.json"
--num_queries 1
--d_model 64
--num_heads 4
--batch_size 128
--seq_max_lens "seq_a:128,seq_b:128,seq_c:256,seq_d:256"
--reinit_sparse_after_epoch 0
--reinit_cardinality_threshold 0
--sparse_lr 0.05
--dropout_rate 0.01
--amp
--amp_dtype bfloat16
```

理由：

- `group` tokenizer 使用 `ns_groups.json` 的 7 个 user groups + 4 个 item groups，加 user dense token 后 `num_ns=12`。
- 设 `num_queries=1` 时，`T = 1*4 + 12 = 16`，仍满足 `d_model=64` 的 RankMixer full 约束。
- 这能直接检验“语义 NS token 是否优于 rankmixer 均分 chunk”，比复刻 `reinit_after=2` 信息增益更高。

备选实验：

```bash
--seq_encoder_type swiglu
--ns_tokenizer_type group
--ns_groups_json "${SCRIPT_DIR}/ns_groups.json"
--num_queries 2
--d_model 80
--num_heads 4
```

此时 `T = 2*4 + 12 = 20`，`80 % 20 == 0`。它同时增加 query 容量和 hidden 维度，但变量更多，建议放在 group+num_queries=1 之后。

### EDA 脚本

新增只读脚本：

```bash
python research/code/inspect_pcvr_structure.py \
  --parquet sample_data/demo_1000.parquet
```

如果有平台 `schema.json`，可进一步分析 skipped fids 和 RankMixer chunk：

```bash
python research/code/inspect_pcvr_structure.py \
  --parquet /path/to/train_data \
  --schema /path/to/schema.json \
  --ns-groups train/ns_groups.json \
  --emb-skip-threshold 1000000 \
  --user-ns-tokens 5 \
  --item-ns-tokens 2
```

sample data 观察：

```text
label_type_counts={1: 876, 2: 124}
seq_a length mean=701, p50=577, p90=1562, max=1888
seq_b length mean=571, p50=405, p90=1393, max=1952
seq_c length mean=449, p50=322, p90=887, max=3894
seq_d length mean=1100, p50=1035, p90=2215, max=3951
```

因此当前 `seq_max_lens=128/128/256/256` 是强截断配置，提速明显，但可能损失长序列信息。后续在 SwiGLU 稳定后，可以单独尝试恢复 `seq_d` 或 `seq_a` 长度。

### 资源观察与调参含义

用户补充的最好实验资源画像：

```text
GPU MEM 约 8000MB
DATA 约 18000MB
CPU 利用率不超过 30%
GPU 利用率较高，约 100%~150%，但有明显波峰和波谷
1 epoch 约 35min
```

含义：

- 当前 `swiglu + batch_size=128 + seq=128/128/256/256` 并没有吃满显存，存在把 batch 或序列长度加回去的空间。
- CPU 利用率不高，主要瓶颈不像是纯 CPU 预处理；GPU 利用率波峰波谷更像 DataLoader/IO、row-group buffer、或不同 batch 序列有效长度差异造成的供给不均。
- 因为 `SwiGLUEncoder` 没有 self-attention 的 `O(L^2)`，序列长度翻倍的风险远小于 Transformer；下一步可以优先试恢复 `seq_d` 或全量恢复默认长度。
- batch size 从 128 回到 256 值得试，尤其在显存只有约 8GB 占用时；但 batch 变大可能降低梯度噪声，官方 eval 未必必然提升，需要单独消融。

### 当前结构问题逐项判断

1. `group` tokenizer 为什么值得试：
   - 当前 `rankmixer + ns_groups_json=""` 不使用语义分组，而是把 46 个 user int fid 按顺序拼接后均分 5 token。
   - user 侧 `46*64/5` 不是整数个 feature embedding，token 边界会切进 fid embedding 中间。
   - `group + ns_groups.json` 会把 user/item fid 按人工语义组聚合，token 更可解释；代价是需要把 `num_queries` 调为 1 才保持 `T=16,d_model=64`。

2. `num_queries` 是否合理：
   - 当前 `rankmixer` 下 `num_ns=8,num_queries=2,T=16`，每个序列域 2 个 query，4 个域共 8 个 query token，和 8 个 NS token 比例为 1:1。
   - `group` 下 `num_ns=12`，若还用 `num_queries=2` 则 `T=20`，`d_model=64` 不合法；所以最小实验必须 `num_queries=1`。
   - `num_queries=1` 会降低序列表达容量，但能验证“语义 NS token 是否重要”。若有效，再上 `d_model=80,num_queries=2,T=20`。

3. `num_heads` 是否要调：
   - 当前 `d_model=64,num_heads=4`，head_dim=16，是稳妥配置。
   - `swiglu` 序列编码器本身不用 heads，heads 只影响 query-to-sequence cross-attention。
   - 在没有确认 query/token 结构前，不优先调 heads；`num_heads=8` 会让 head_dim=8，未必更好。

4. `num_hyformer_blocks` 是否太少：
   - 2 blocks 已经能把 valid/eval 推高，说明不是明显欠拟合。
   - 增到 3 会增加 cross-attention/FFN 深度，可能提升表达，也可能加重过拟合和训练时间。
   - 建议放在 batch/seq/ns tokenizer 后面。

5. `dropout=0.03` 的当前判断：
   - 之前 `dropout=0.1 + sparse_lr=0.01` 明显削弱有效信号。
   - 现在 full reinit 已经是强正则，`dropout=0.01` 表现更好；`0.03` 可以试，但不再是最高优先级。

6. `emb_skip_threshold`：
   - 官方/原版默认 `0` 表示不跳过任何 embedding。
   - 当前 `1000000` 会跳过 seq_b 1 个、seq_c 3 个高基数字段，这可能省显存，也可能丢掉关键 ID 信号。
   - 但 `emb_skip_threshold=0` 已验证不可行：总参数约 `10,341,768,961`，sparse params 约 `10,339,307,392`，在 Adagrad 初始化同尺寸 `sum` state 时 OOM。
   - 后续不能再直接试 0，应改为阈值搜索，例如 `2000000 / 5000000 / 7000000`，逐步恢复部分高基数字段；如果 7M 或更高阈值已经 optimizer OOM，就把 5M 视为当前可行上界。

### `emb_skip_threshold=0` 失败结论

日志定位：

```text
torch.optim.Adagrad.__init__
state["sum"] = torch.full_like(...)
torch.OutOfMemoryError: CUDA out of memory
```

这不是 forward/backward OOM，而是 optimizer 初始化 OOM。原因是 Adagrad 会为每个 sparse parameter 维护同尺寸累计平方梯度 `sum`。

参数量对比：

```text
emb_skip_threshold=1000000:
Total parameters: 239,897,345
Sparse params: 237,435,776

emb_skip_threshold=0:
Total parameters: 10,341,768,961
Sparse params: 10,339,307,392
```

结论：

- `0` 在当前代码语义中是“不启用 skip”，不是“跳过所有特征”。
- 全量恢复高基数 embedding 参数量约放大 43 倍，且 Adagrad state 再额外翻倍，当前平台不可行。
- 下一轮应试 `emb_skip_threshold=5000000`；若 optimizer 初始化仍 OOM，退到 `2000000`。

### 2026-04-30 两组结构实验结果

| 实验 | 关键配置 | 参数量 | Best valid | 官方 eval AUC | 判断 |
|---|---|---:|---:|---:|---|
| RankMixer + threshold 5M | `swiglu, rankmixer, num_queries=2, emb_skip_threshold=5000000` | 305.9M | 0.861063@epoch6 | 0.809921 | 当前最佳；恢复部分高基数序列 ID 对官方 eval 有收益 |
| Group NS | `swiglu, group, num_queries=1, emb_skip_threshold=1000000` | 239.5M | 0.861189@epoch7 | 0.805905 | valid 高但官方 eval 低；不作为当前主线 |

RankMixer + 5M 的 skipped 情况：

```text
emb_skip_threshold=5000000:
seq_b skipped 1/13 features
seq_c skipped 2/11 features
```

对比 1M 时 `seq_c skipped 3/11`，5M 至少恢复了一个 seq_c 高基数字段。官方 eval 从 `0.806956` 提到 `0.809921`，说明被恢复的高基数字段含有测试集有效信号。

Group NS 实验解读：

- 它验证的是 `ns_groups.json` 语义分组，但为了满足 `d_model=64` 的 `T=16` 约束，`num_queries` 从 2 降到 1。
- 因此它同时改变了 NS token 语义和序列 query 容量。
- valid AUC 最高但官方 eval 低，说明当前不能用 valid AUC 单独选择模型；也说明 group+query=1 暂时不适合作为冲分主线。
- 若后续继续 group，应试 `d_model=80,num_queries=2,num_heads=4`，让 `T=2*4+12=20` 且 `80%20==0`，但变量更多，优先级低于阈值/序列长度搜索。

### 同事 Notebook 带来的新启发

用户补充了 `all_NOTEBOOK.py`、`dataset_full_walkthrough.ipynb`、`model_full_walkthrough.ipynb`。其中 `all_NOTEBOOK.py` 主要是 dataset walkthrough 的导出；两个 notebook 对后续冲分有以下启发：

1. 数据侧：
   - `label = (label_type == 2)`，其余 label_type 都被当作负样本。若正式数据存在 `0/1/2` 等多种负样本类型，后续可以做 hard/easy negative 权重，但当前没有足够证据优先改 loss。
   - train/valid 按 Row Group 切分，训练只做 `buffer_batches` 窗口 shuffle，不是全量随机。valid 与官方 test 相关性弱是合理现象，官方 eval 权重应高于 valid 排名。
   - `{domain}_time_bucket` 已经在 dataset 侧生成，并在模型侧作为时间 embedding 加到序列 token。时间特征工程不是空白，但直接改 bucket 边界会改变所有序列时间表示，优先级低于阈值/序列容量。

2. 模型侧：
   - `MultiSeqQueryGenerator` 的输入是 `NS_flat + seq_mean`，为每个序列域生成独立 Q tokens。`num_queries` 不是普通小参数，它直接决定每个序列域能向 HyFormer 带入多少摘要视角。
   - `RankMixerBlock` 要求 `rank_mixer_mode=full` 下 `d_model % T == 0`，其中 `T = num_queries * num_sequences + num_ns`。这解释了为什么 group 实验不能简单保留 `num_queries=2,d_model=64`。
   - `LongerEncoder` 是当前最值得从 notebook 迁移成实验变量的模块：`L > top_k` 时用最近 top-k token 做 Q，对完整序列 cross-attention 压缩；后续 block 在 top-k 上 self-attention。它比 Transformer 更适合超长序列，比 SwiGLU 多了序列 token 交互。

3. 对 group 实验的修正：
   - `group + num_queries=1` 不是纯 NS token 分组消融，因为它同时把每个序列域的 query 容量从 2 降到了 1。
   - 若后续公平验证 group，应使用 `d_model=80,num_queries=2,num_heads=4`，此时 `num_ns=12,T=20,80%20==0`。
   - 由于 group 官方 eval `0.805905` 低于 rankmixer 5M 的 `0.809921`，当前不把 group 作为主线。

### 下一步两卡建议（2026-04-30 更新）

若用户提到的 `7000000` 或更高阈值已经 OOM，则不要继续硬追阈值；`5000000` 暂时是当前可行上界和官方 eval 最好点。下一轮两卡从“恢复更多信息”切到“更好利用已恢复的信息”。

**卡 1：RankMixer 5M + 恢复默认长序列**

```bash
--seq_encoder_type swiglu
--ns_tokenizer_type rankmixer
--ns_groups_json ""
--user_ns_tokens 5
--item_ns_tokens 2
--num_queries 2
--d_model 64
--num_heads 4
--emb_skip_threshold 5000000
--batch_size 128
--seq_max_lens "seq_a:256,seq_b:256,seq_c:512,seq_d:512"
--num_epochs 8
--patience 4
--sparse_lr 0.05
--dropout_rate 0.01
--reinit_sparse_after_epoch 0
--reinit_cardinality_threshold 0
--amp
--amp_dtype bfloat16
```

理由：sample data 显示四个序列域都远长于 `128/128/256/256`，当前短序列是为速度/显存做的强截断。SwiGLU 是 O(L) 编码器，恢复长度比 Transformer 安全。

**卡 2：LongerEncoder 5M + 长序列**

```bash
--seq_encoder_type longer
--seq_top_k 64
--seq_causal false
--ns_tokenizer_type rankmixer
--ns_groups_json ""
--user_ns_tokens 5
--item_ns_tokens 2
--num_queries 2
--d_model 64
--num_heads 4
--emb_skip_threshold 5000000
--batch_size 128
--seq_max_lens "seq_a:256,seq_b:256,seq_c:512,seq_d:512"
--num_epochs 8
--patience 4
--sparse_lr 0.05
--dropout_rate 0.01
--reinit_sparse_after_epoch 0
--reinit_cardinality_threshold 0
--amp
--amp_dtype bfloat16
```

OOM 退路：

- optimizer 初始化 OOM：回到 `emb_skip_threshold=1000000`，说明 5M + longer 组合参数/状态太重。
- forward/backward OOM：先 `seq_top_k=32`，再 `batch_size=64`。

暂不建议：

- `--torch_compile`：已有经验是可能掉 AUC，冲分阶段收益不稳定。
- `dropout=0.03`：可做小消融，但 full reinit 已经是强正则，不应抢在长序列/LongerEncoder 前面。
- `num_hyformer_blocks=3`：结构更深但变量更大，应等长序列和 longer 结果出来后再试。

### 2026-05-01 反馈：长序列 valid 提升但官方不涨

用户在 2026-05-01 提供了两组新训练日志，其中 `5M + seq*2` 已提交官方 eval：

| 实验 | 配置 | 参数量 | Best valid | 官方 eval | 推理耗时 |
|---|---|---:|---:|---:|---:|
| 5M short seq | `swiglu, threshold=5M, seq=128/128/256/256` | 305.9M | 0.861063@epoch6 | 0.809921 | 129.65s |
| 5M seq*2 | `swiglu, threshold=5M, seq=256/256/512/512` | 305.9M | 0.862731@epoch6 | 0.809907 | 232.36s |
| 7M short seq | `swiglu, threshold=7M, seq=128/128/256/256` | 674.8M | 0.861692@epoch7 | 0.806861 | 待补 |

#### 长序列实验结论

`seq*2` 训练曲线是健康的：

```text
epoch1 valid AUC=0.858060
epoch4 valid AUC=0.861789
epoch5 valid AUC=0.862448
epoch6 valid AUC=0.862731, LogLoss=0.223918, Brier=0.064390
epoch8 valid AUC=0.862651
```

但官方 eval 为 `0.809907`，略低于 `5M short seq` 的 `0.809921`，且推理耗时从 `129.65s` 增至 `232.36s`。因此：

- 长序列信息改善了当前 valid split，但没有转化为官方 test 收益。
- 这进一步证明 valid/test 分布不一致是当前主要瓶颈之一。
- 在官方 eval 有限的情况下，不应继续为 `seq*2` 额外提交更多 epoch，除非后续 dataset 对齐分析证明 test 更偏长序列。

#### 7M 阈值实验结论

7M 配置没有 OOM，但参数量明显变大：

```text
5M total params: 305,900,929
7M total params: 674,819,905
7M sparse params: 672,358,336
7M best valid AUC: 0.861692@epoch7
```

相对 5M，7M 多恢复了一个超高基数字段级别的 embedding，参数量约 2.2 倍，但 valid 没有明显收益，官方 eval 也降到 `0.806861`。结论：继续放宽 `emb_skip_threshold` 已经不再是主线，固定 `5000000`。

#### 数据侧优先级上升

用户判断“如果数据可以解决验证集和测试集不一致，后面优化方向会容易很多”是对的。下一步应优先补 dataset 诊断，而不是继续堆模型：

1. 对 train/valid/test 三侧分别统计序列长度分布、空序列比例、time bucket 分布。
2. 统计 `clip_vocab` 触发比例和各序列域高基数字段覆盖/OOV 情况，确认 5M 恢复的字段是否更贴近 test。
3. 做 Row Group 级别 valid 稳定性分析：不同 row group 区间的 label_mean、seq_len、time bucket 是否漂移。
4. 若正式 test 无 label，只做特征分布对齐；目标是找到更像 test 的 valid split 或加权指标，而不是盲目追当前 tail-valid AUC。

#### 2026-05-01 晚两卡建议

7M 官方下降后，今晚不再继续 threshold 或 seq length。用户提醒“必要时可以改代码”，因此新增最小数据切分开关 `--valid_split_mode {tail,head,middle,random,time_tail}`：默认仍是 `tail`，兼容旧实验；新模式通过显式 RowGroup indices 让 train/valid 可以不再是单一连续尾部切分。

用户补充外部经验：有人按时间划分 validation，且有人观察到官方 test 可能集中在周日/周一的 23/00/01 附近。随机 RowGroup valid 可能破坏时间外推关系，不适合作为冲分默认。因此把新增切分开关扩展为 `--valid_split_mode time_tail`：按 Row Group 的最大 timestamp 排序后取最新 Row Groups 做 valid，并打印 train/valid timestamp range。

若 time-tail 官方 eval 优于历史 tail-valid，则后续所有结构、loss、特征工程实验都应使用同一 time-tail 协议；否则继续使用历史 tail-valid。这样可以避免每个实验都在不同 validation 定义下选 checkpoint。`random` 保留为诊断模式，不作为主实验协议。

训练/评估同步边界：

- `valid_split_mode` 只影响训练时 train/valid RowGroup 切分和 best checkpoint 选择，不进入官方 test 推理链路，因此本次不需要改 `eval/dataset.py`。
- 改模型结构时必须同步 `train/model.py` 与 `eval/model.py`。
- 改输入特征构造、time bucket、clip/OOV 逻辑时必须同步 `train/dataset.py` 与 `eval/dataset.py`。
- 改训练策略、optimizer、sampling、valid split 通常只影响 `train/`。

两张卡分别做一个数据侧变量和一个结构侧变量，并且 group 使用同样的 time-tail：

**卡 1：5M short seq + time-tail RowGroup valid**

```bash
--seq_encoder_type swiglu
--ns_tokenizer_type rankmixer
--ns_groups_json ""
--user_ns_tokens 5
--item_ns_tokens 2
--num_queries 2
--d_model 64
--num_heads 4
--emb_skip_threshold 5000000
--batch_size 128
--seq_max_lens "seq_a:128,seq_b:128,seq_c:256,seq_d:256"
--valid_ratio 0.1
--valid_split_mode time_tail
--seed 42
--num_epochs 8
--patience 4
--sparse_lr 0.05
--dropout_rate 0.01
--reinit_sparse_after_epoch 0
--reinit_cardinality_threshold 0
--amp
--amp_dtype bfloat16
```

目的：不改模型，仅改变 checkpoint 选择所依赖的 valid 分布。若官方提升，说明历史文件尾部 valid 不够贴近时间切片 test，后续应继续做 timestamp、feature drift 和 OOV/clip 诊断。

**卡 2：公平 semantic group + query=2**

```bash
--seq_encoder_type swiglu
--ns_tokenizer_type group
--ns_groups_json "${SCRIPT_DIR}/ns_groups.json"
--num_queries 2
--d_model 80
--num_heads 4
--emb_skip_threshold 5000000
--batch_size 128
--seq_max_lens "seq_a:128,seq_b:128,seq_c:256,seq_d:256"
--valid_ratio 0.1
--valid_split_mode time_tail
--seed 42
--num_epochs 8
--patience 4
--sparse_lr 0.05
--dropout_rate 0.01
--reinit_sparse_after_epoch 0
--reinit_cardinality_threshold 0
--amp
--amp_dtype bfloat16
```

目的：修正旧 group 实验的混杂变量。旧实验是 `group + d_model=64 + num_queries=1`，同时改变了 NS token 语义和序列 query 容量；这次 `num_ns=12,T=2*4+12=20,d_model=80`，保留每个序列域 2 个 query，并使用同一 time-tail 选模协议。如果仍低于 5M RankMixer 主线，才更能说明当前 `ns_groups.json` 人工分组不如 RankMixer 分块。

备选：如果今晚只追最稳 leaderboard，不想跑结构变量，则把卡 2 改为同一主线换种子（例如 `--seed 2026`），因为用户提到实战经验里“找好种子”也可能带来显著收益。`LongerEncoder` 暂时排在公平 group 之后，因为用户明确希望重新审视语义分组，且 group 上一轮存在 query 容量混杂。

#### 每日额度策略（2026-05-02 修正）

用户查阅官方规则后修正：每日提交次数上限只针对官方评估任务，每 24 小时重置一次，重置时间为游戏时间 15:59:59；每个团队每个连续 24 小时内最多提交 3 个评估任务，失败或已停止任务不计入。因此训练可以多开，真正稀缺的是官方 eval。

若当天剩余 2 次 eval 额度，仍应优先保留 1 次给当前最关键 checkpoint：

```text
训练可以多跑；官方 eval 只提交最值得验证的 checkpoint
```

优先官方 eval `5M short seq + time-tail RowGroup valid`，因为它决定后续所有实验是否换 validation 协议。若今晚额外开训练，先看训练日志再决定是否占用第二次 eval。

#### 平台代码更新清单

以后每个实验配置都要同时说明平台要更新哪些文件：

- `time_tail valid`：更新训练侧 `train/dataset.py`、`train/train.py` 和 `run.sh`/`script_args`；官方 eval 侧不用改。
- 公平 group：同上，且平台训练包要包含 `train/ns_groups.json`；官方 eval 侧不用改，但 checkpoint sidecar 必须包含 `train_config.json/schema.json/ns_groups.json`。
- 模型结构代码变更：同步 `train/model.py` 和 `eval/model.py`。
- 输入特征、时间特征、OOV/clip 逻辑变更：同步 `train/dataset.py` 和 `eval/dataset.py`。
- loss、optimizer、warmup、reinit、valid split：通常只改 `train/`。

#### 今晚额外训练：公平 group + time-tail + focal

用户希望开一个“语义分组 + 高分经验”的混合训练。推荐配置：

```bash
--seq_encoder_type swiglu
--ns_tokenizer_type group
--ns_groups_json "${SCRIPT_DIR}/ns_groups.json"
--num_queries 2
--d_model 80
--num_heads 4
--emb_skip_threshold 5000000
--batch_size 128
--seq_max_lens "seq_a:128,seq_b:128,seq_c:256,seq_d:256"
--valid_ratio 0.1
--valid_split_mode time_tail
--seed 42
--num_epochs 8
--patience 4
--sparse_lr 0.05
--dropout_rate 0.01
--loss_type focal
--focal_alpha 0.5
--focal_gamma 2.0
--reinit_sparse_after_epoch 0
--reinit_cardinality_threshold 0
--amp
--amp_dtype bfloat16
```

取舍：

- 这是冲分型混合实验，不是干净消融。
- 选择 focal 是因为代码已支持，不需要改 eval；相比 warmup/scale、pair 特征、绝对 hour/week，它今晚落地风险最低。
- 不加裸 hour/week：外部经验提示它可能掉点，且可能放大 test 特定时间窗口的分布偏差。
- 不临时做 pair 特征/item 特征新构造：这需要同步 train/eval dataset，适合等 time-tail 协议结果回来后再做。

是否提交官方 eval：

- 若 focal 让 LogLoss/校准明显变坏，即使 valid AUC 高也谨慎提交。
- 若 valid AUC 接近或高于当前 5M 主线，且 `prob_mean` 接近 `label_mean`、LogLoss 未明显恶化，可考虑用第二次 eval。

#### 高分经验纳入第三优先级

别人 `0.824` 级别经验值得后续系统学习，但应放在 time-tail 和公平 group 之后：

1. `scale + lr warmup`：可能是优化器侧有效技巧；需要明确 scale 作用在 logit、embedding 还是特征归一化上。
2. 特征处理：最可能带来大收益，但一旦改输入构造，必须同步 train/eval dataset。
3. `focal loss`：当前代码已有 `loss_type=focal`，但默认 `focal_alpha=0.1` 对正样本可能偏低，需单独设计参数。
4. 时间特征：当前已有“序列行为 timestamp 相对曝光 timestamp 的 time-delta bucket”。外部经验提示 test 可能集中在周日/周一夜间窗口；直接加 hour/week bucket 有人掉点，说明绝对时间特征可能放大分布偏差。后续更优先做序列行为时间戳的 recency/间隔特征，或先确认 time-tail valid 是否更贴近 test。
5. 不建议重复：关闭 sparse reinit、batch 内 resample。前者已在本项目掉点，后者外部经验也掉点。
## 2026-05-02 新结果：valid 协议与 focal 混合实验复盘

### time-tail RowGroup valid 未能贴近官方 test

用户在 2026-05-02 提供了 `5M short seq + valid_split_mode=time_tail` 的官方 eval：

```text
official eval AUC = 0.806627
best valid AUC = 0.862431@epoch6
config = rankmixer + swiglu + emb_skip_threshold=5000000 + BCE + full reinit
```

该结果低于历史主线 `RankMixer 5M short seq + tail-valid` 的 `0.809921`。因此，不能把 `time_tail` 升级为后续默认验证协议。日志中的 train/valid timestamp range 仍然高度重叠，说明按 RowGroup `max(timestamp)` 排序取尾部，并没有构造出真正的时间外推验证集。当前更合理的解释是：官方 test 可能是更特殊的时间/星期/小时切片，而不是简单的 RowGroup 尾部。

后续若要继续解决 valid/test 不一致，优先做数据诊断而不是继续换 split 名称：

- 打印 train/valid/test 的 timestamp、hour、weekday 分布；
- 打印 RowGroup 级别时间范围是否高度重叠；
- 比较 item/user/pair 特征在 train 与 test 的分布漂移；
- 比较序列长度、序列 recency bucket、OOV/clip/skip 覆盖率；
- 明确哪些特征处理改动需要同步 `train/dataset.py` 与 `eval/dataset.py`。

### semantic group + focal 是混合实验，不应提交 eval

用户还提供了 `semantic group + d_model=80 + num_queries=2 + time_tail + focal_alpha=0.5` 的训练日志：

```text
best valid AUC = 0.861039@epoch5
valid label_mean ≈ 0.096731
valid prob_mean ≈ 0.234 - 0.252
valid LogLoss ≈ 0.320 - 0.337
```

虽然 AUC 不算崩，但概率校准已经明显坏掉：预测均值约为真实正样本率的 2.5 倍，LogLoss/Brier 远差于 BCE 主线。因此这次不建议占用官方 eval。它也不能作为“semantic group 不行”的证据，因为变量混在了一起：`group`、`d_model=80`、`time_tail`、`focal_alpha=0.5` 同时变化。

后续拆分优先级：

1. 若验证 semantic group：用 `BCE + tail-valid + d_model=80 + num_queries=2 + threshold=5M`，不要混 focal。
2. 若验证 focal：回到 RankMixer 5M 主线，单独改 loss；优先小 alpha，不再默认 `alpha=0.5`。
3. 若参考高分经验：优先 item 侧特征、pair 特征、序列行为时间戳的相对 recency/interval 特征、优化器 warmup/scale。
4. 暂停绝对 hour/week 裸特征，除非先通过分布诊断证明它不会放大 test 切片偏差。

当前主线仍然是：

```text
rankmixer + swiglu + emb_skip_threshold=5000000
seq_a/b/c/d = 128/128/256/256
BCE + sparse_lr=0.05 + dropout=0.01
full sparse reinit: reinit_after=0, threshold=0
historical tail-valid
official eval AUC = 0.809921
```
## 2026-05-02 Transformer 5M 与官方 test 分布诊断

用户进一步验证了 `Transformer + emb_skip_threshold=5000000 + short seq + full reinit`：

```text
official eval AUC = 0.80656
best valid AUC = 0.860758@epoch4
```

这组实验与 `SwiGLU + 5M` 是干净对照，说明在当前主线上不应回退 Transformer：

```text
SwiGLU + 5M official = 0.809921
Transformer + 5M official = 0.80656
```

更关键的是 eval 侧分布诊断第一次输出了官方 test 的真实时间形态：

```text
samples = 310000
timestamp_span_hours = 1.55
top UTC hours = 0:65.42%, 23:20.91%, 1:13.67%
top CN hours = 8:65.42%, 7:20.91%, 9:13.67%
top CN weekdays = Mon:100.00%
```

结论：官方 test 是 100% 周一、约 1.55 小时的窄时间切片。此前 `tail` / `time_tail` RowGroup valid 的 timestamp range 都远宽于 test，且 train/valid timestamp range 高度重叠；因此 valid/test 不一致已经不是抽象问题，而是具体的时间切片问题。

序列长度诊断：

```text
seq_a len_avg=120.666, max=128
seq_b len_avg=111.716, max=128
seq_c len_avg=225.077, max=256
seq_d len_avg=251.505, max=256
```

test 上序列几乎贴近当前截断上限，但 `seq*2` 官方没有涨，说明问题不是简单“长度不够”，而是如何利用序列时间、recent/interval、item/pair 关系。后续不应继续盲目加长序列，而应先做 test-aware 分布对齐和更有针对性的时间/特征处理。

## 2026-05-02 样本级 time-window valid

在确认官方 test 是 100% 周一、约 1.55 小时窄窗口后，RowGroup 级 `tail/time_tail` 已不足以模拟 test。新增训练侧切分：

```text
--valid_split_mode time_window
--valid_time_window_hours 2.0
```

语义：

```text
cutoff_ts = train_global_max_timestamp - valid_time_window_hours * 3600
train = samples with timestamp < cutoff_ts
valid = samples with timestamp >= cutoff_ts
```

这个改动只影响训练阶段的 train/valid 切分和 best checkpoint 选择；官方 eval 推理不需要同步。平台训练需要更新 `train/dataset.py` 和 `train/train.py`。

第一组建议仍保持当前最强结构不变：

```text
rankmixer + swiglu + threshold=5M + short seq + BCE + full reinit
valid_split_mode=time_window
valid_time_window_hours=2.0
```

该实验的目标不是单纯追 valid AUC，而是确认“样本级窄时间窗口 valid”是否比 RowGroup tail 更能解释 official eval。如果仍然不能解释，后续需要从特征层面对齐 test：序列 recency/interval、item 侧特征、pair 特征，以及优化器 warmup/scale。

## 2026-05-03 time-window valid 反馈

`time_window=2h` 官方 eval AUC 为 `0.7883`，显著低于主线 `0.809921`。

关键诊断：

```text
valid_time_window_hours = 2.0
valid_num_samples = 322703
best valid AUC = 0.820746@epoch6
train label_mean ≈ 0.088
valid label_mean = 0.113259
official test begins at timestamp_min = 1774222835
```

这说明样本级 time-window 更像 test，但作为训练协议会扣掉最靠近官方 test 的有标签样本。官方 test 是紧贴训练最大 timestamp 之后的 1.55 小时窗口，最近 2 小时训练样本对泛化到 test 很可能非常关键；把它们全部作为 valid 会伤害最终模型。

因此更新结论：

- `time_window` 保留为诊断工具；
- 不应直接用 `time_window` 训练出的 checkpoint 冲榜；
- 冲榜模型应尽量使用全量近邻时间数据；
- 下一步从 test-aware 特征处理入手：序列 recency/interval、item 侧特征、pair 特征，而不是继续换 holdout 切分。
## 2026-05-03 latest experiment ledger and TODO

### What to submit next

Submit the optimizer-direction checkpoint first:

```text
candidate = RankMixer + SwiGLU + 5M + short seq + BCE + full reinit
only changed knob = sparse_lr 0.05 -> 0.08
best valid AUC = 0.861314@epoch6
valid LogLoss = 0.224980
valid prob_mean = 0.096718
valid label_mean = 0.096785
official eval = pending
```

Reason: this is the healthiest pending run. It did not inflate probability mean like focal loss, and it directly tests the external advice that optimizer / learning-rate details matter.

### Do not spend eval on these without a new reason

```text
focal_alpha=0.25, gamma=2:
  best valid AUC = 0.861691@epoch8
  valid prob_mean ~= 0.17 - 0.18
  valid label_mean = 0.096785
  valid LogLoss ~= 0.273 - 0.285
  conclusion = AUC is acceptable, calibration is badly high; skip official eval for now.

use_seq_time_delta_proj=True:
  official eval AUC = 0.809214
  conclusion = this specific token-level scalar time-delta projection did not beat 0.809921.

train_include_valid=True + checkpoint_select_metric=last:
  official eval AUC = 0.80387
  valid AUC = 0.889893, inflated because validation rows are in training.
  conclusion = not a useful checkpoint-selection path.

time_window=2h:
  official eval AUC = 0.7883
  conclusion = removes the closest training timestamps; keep only as distribution diagnostic.
```

### External high-score experience to re-read before next code edit

The useful clues from others are still plausible, but our first implementations were probably too literal:

- `scale + lr warmup`: next optimizer direction after `sparse_lr=0.08` should be warmup or scale, not random architecture growth.
- item-side features: official eval diagnostics show high zero rates, especially user side; item / item-context signals may be more stable than user ID memorization.
- pair features: PCVR is interaction-driven, so user-item / item-sequence compatibility features may be more valuable than simply adding model depth.
- sequence time features: hour / week buckets alone can hurt, and scalar per-token delta projection did not help. The better direction is aggregate recency / interval / recent-window statistics per sequence domain.
- focal loss: the current alpha is too aggressive. If we revisit focal, use lower alpha and judge calibration before spending eval quota.

### Current rollback baseline

```text
official eval AUC = 0.809921
seq_encoder_type = swiglu
ns_tokenizer_type = rankmixer
emb_skip_threshold = 5000000
seq_max_lens = seq_a:128,seq_b:128,seq_c:256,seq_d:256
loss_type = bce
sparse_lr = 0.05
dropout_rate = 0.01
full sparse reinit = reinit_sparse_after_epoch 0 + reinit_cardinality_threshold 0
valid_split_mode = tail
checkpoint_select_metric = auc
```
