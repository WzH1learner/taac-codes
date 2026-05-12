# TAAC 2026 冲分 Memory：D01 新主线与下一步决策

更新时间：2026-05-12  
用途：放入 Codex 可读取目录，作为后续 TAAC 提分任务的统一上下文。  
建议路径：`TAAC/docs/3-实验记录/MEMORY_0512_D01_NEW_BEST.md` 或 Codex 项目 memory 文件。  

---

## 0. 当前最重要结论

最新三次 official eval 结果：

| 实验 | official eval AUC | 结论 |
|---|---:|---|
| E00 | 0.808496 | 不是严格复现历史 best，因为实际跑的是 `transformer`，不是文档里历史 best 的 `swiglu` |
| T01 | 0.801715 | 当前 time_context 实现强失败，禁止组合 |
| D01 | **0.818095** | 当前新 best，主线切换到 grouped user_dense |

新的当前最佳：

```text
official eval AUC = 0.818095
experiment = D01_grouped_user_dense_single_token
seq_encoder_type = transformer
ns_tokenizer_type = rankmixer
emb_skip_threshold = 5000000
seq_max_lens = seq_a:128,seq_b:128,seq_c:256,seq_d:256
loss_type = bce
sparse_lr = 0.05
dropout_rate = 0.01
user_dense_projector_type = grouped
use_time_context = 0
rank_mixer_mode = full
num_queries = 2
user_ns_tokens = 5
item_ns_tokens = 2
d_model = 64
num_heads = 4
reinit_sparse_after_epoch = 0
reinit_cardinality_threshold = 0
checkpoint_select_metric = auc
```

旧的 `official eval AUC = 0.809921` 只能作为 old fallback，不再是当前 best。

---

## 1. 关键纠偏：E00 不是严格复现

历史文档中写的旧 best 是：

```text
official eval AUC = 0.809921
seq_encoder_type = swiglu
ns_tokenizer_type = rankmixer
emb_skip_threshold = 5000000
loss_type = bce
short seq
```

但最新三次日志显示，E00/T01/D01 实际配置里：

```text
seq_encoder_type = transformer
```

原因是：`train/run.sh` 没有显式传 `--seq_encoder_type swiglu`，而 `train/train.py` 的默认值是 `transformer`。

因此：

```text
E00 = transformer + flat user_dense + no time_context
T01 = transformer + flat user_dense + time_context
D01 = transformer + grouped user_dense + no time_context
```

这三次实验不能直接和“swiglu old best”做严格等价复现实验对比。

---

## 2. 三次实验复盘

### 2.1 E00_reproduce_080992

官方结果：

```text
official eval AUC = 0.808496
```

关键日志配置：

```text
seq_encoder_type = transformer
user_dense_projector_type = flat
use_time_context = 0
emb_skip_threshold = 5000000
sparse_lr = 0.05
loss_type = bce
rankmixer, T=16, d_model=64
```

训练日志：

```text
best epoch = 5
best valid AUC = 0.860380
valid LogLoss at best epoch ≈ 0.226409
valid prob_mean at best epoch ≈ 0.086385
valid label_mean = 0.096785
```

结论：

```text
E00 不是严格复现 0.809921，因为它实际跑 transformer。
它可以作为 transformer + flat dense 的对照基线。
```

---

### 2.2 T01_time_context

官方结果：

```text
official eval AUC = 0.801715
```

关键日志配置：

```text
seq_encoder_type = transformer
user_dense_projector_type = flat
use_time_context = 1
emb_skip_threshold = 5000000
sparse_lr = 0.05
loss_type = bce
rankmixer, T=16, d_model=64
```

训练日志：

```text
best epoch = 4
best valid AUC = 0.860832
valid LogLoss at best epoch ≈ 0.225411
valid prob_mean at best epoch ≈ 0.089270
valid label_mean = 0.096785
```

结论：

```text
当前 time_context 接入方式 official 强失败。
不要继续：
- D01 + T01
- T01 + hash
- T01 + swiglu
- T01 + any combo
```

补充说明：

时间方向在历史队友文档里也已经多次表现弱，例如：
- `time_window=2h` = 0.7883
- `use_seq_time_delta_proj=True` = 0.809214
- 当前 T01 = 0.801715

因此当前阶段 time_context / 时间特征方向降为低优先级，除非先做新的机制诊断。

---

### 2.3 D01_grouped_user_dense_single_token

官方结果：

```text
official eval AUC = 0.818095
```

关键日志配置：

```text
seq_encoder_type = transformer
user_dense_projector_type = grouped
use_time_context = 0
emb_skip_threshold = 5000000
sparse_lr = 0.05
loss_type = bce
rankmixer, T=16, d_model=64
```

训练日志：

```text
best epoch = 6
best valid AUC = 0.864399
valid LogLoss at best epoch ≈ 0.224305
valid prob_mean at best epoch ≈ 0.093914
valid label_mean = 0.096785
```

相对 E00 的增益：

```text
E00 transformer + flat dense = 0.808496
D01 transformer + grouped dense = 0.818095
delta = +0.009599
```

结论：

```text
D01 是当前新 best。
grouped user_dense 是当前已验证最有效提分点。
后续主线围绕 D01 做干净单变量消融和组合，不要回到旧 flat dense。
```

---

## 3. 为什么 D01 有效：结合云端 EDA

云端真实 schema/data 的 user_dense 分布显示：

| fid | dim | 形态判断 |
|---:|---:|---|
| 61 | 256 | UE-like 预训练 embedding，L2 接近 1 |
| 62 | 6 | 百万/千万级 raw count/stat，zero_rate 高 |
| 63 | 19 | 百万/千万级 raw count/stat，zero_rate 高 |
| 64 | 26 | 百万/千万级 raw count/stat，zero_rate 高 |
| 65 | 111 | 百万/千万级 raw count/stat，zero_rate 高 |
| 66 | 150 | 百万/千万级 raw count/stat，zero_rate 高 |
| 87 | 320 | UE-like / large dense embedding，zero_rate 高 |
| 89 | 10 | normalized dense |
| 90 | 10 | normalized dense |
| 91 | 10 | normalized dense |

旧 flat projector 的问题：

```text
所有 user_dense concat -> 一个 Linear -> 一个 user_dense token
```

这个会把 UE-like embedding、normalized dense、百万/千万级统计值混在一起，导致强信号被压坏。

D01 grouped projector 的有效性说明：

```text
fid-aware 分组 + stats log1p/clamp/LayerNorm + 分支投影 + 融合回 1 个 token
显著提升线上 official eval。
```

注意：

队友文档里的旧 `UE split v1 = 0.808142` 不能否定当前 D01。当前 D01 是新的 grouped projector，official 已经到 0.818095。

---

## 4. 队友交接文档中的已知结论

### 4.1 当前应禁用或降级的方向

| 方向 | official eval | 结论 |
|---|---:|---|
| sparse_lr=0.08 | 0.805707 | 禁用，不再提交 |
| dense warmup v1 | 0.807503 | 暂停，不继续 |
| DIN / target-history attention v1 | 0.807927 | 暂停，不直接做 DIN v2 |
| UE split v1 | 0.808142 | 旧实现失败，不作为 D01 的反证 |
| item-pair stats v1 | 未提交 | valid 弱，不建议 official |
| time_window=2h | 0.7883 | 禁用 |
| use_seq_time_delta_proj=True | 0.809214 | 不继续 |
| seq*2 | 0.809907 | 持平但推理慢，不优先 |
| emb_skip_threshold=7M | 0.806861 | 禁用 |
| focal alpha 0.25/0.5 | 未建议提交 | 校准差，不优先 |
| Transformer + 1M + EMA | 0.802788 | 禁用 |
| old NS token boundary fix v1 | 未提交 | valid 不突出，不优先 |

### 4.2 仍可作为后续候选的方向

1. **D01 后续消融**：当前最高优先级。
2. **D02：D01 + swiglu**：只改 `--seq_encoder_type swiglu`，验证 grouped dense 是否与 swiglu 叠加。
3. **D03：D01 + hash embedding clean ablation**：hash v1 接近旧 best，可能和 D01 有叠加，但必须干净单变量。
4. **D01 稳定性复跑 / seed / checkpoint 选择诊断**：0.818 是大涨，需要确认不是一次性偶然。
5. **valid/official mismatch 诊断**：多次 valid 健康但 official 掉，仍需要建设更可靠的筛选协议。

---

## 5. 当前主线重排

新的推进顺序：

```text
P0: 固化 D01 新 best 文档和 checkpoint
P1: 修复 run.sh 显式配置，避免再被默认 transformer/swiglu 漂移污染
P2: 跑 D02 = D01 + swiglu 单变量
P3: 如果 D02 不如 D01，则保留 transformer + grouped dense 为主线
P4: 在 D01 主线上做 clean hash 单变量
P5: 做 D01 稳定性复跑或 seed/checkpoint 诊断
```

不要再按照旧蓝图继续把 `sparse_lr=0.08 / warmup / DIN / UE split / time_context` 作为优先方向。

---

## 6. 必须更新的项目文档

请 Codex 更新以下文件：

```text
TAAC/docs/EXPERIMENT_TRACKING.md
TAAC/docs/3-实验记录/当前实验交接.md
TAAC/docs/1-核心蓝图/推荐算法优化蓝图.md
```

必须写入：

```text
Current Best = 0.818095
Current Best Config = transformer + rankmixer + 5M + BCE + short seq + grouped user_dense + no time_context
Old Best = 0.809921 swiglu/rankmixer old fallback
E00 not strict reproduction because seq_encoder_type defaulted to transformer
T01 rejected
D01 verified effective
Next = D02_grouped_dense_swiglu
```

---

## 7. run.sh 必须修复

问题：

当前 `train/run.sh` 没有显式写 `--seq_encoder_type`，导致实验依赖 `train.py` 默认值。

要求：

所有 active block 和 experiment block 必须显式写出：

```bash
--seq_encoder_type transformer
```

或：

```bash
--seq_encoder_type swiglu
```

不能再依赖默认值。

当前新 best active block 应为：

```bash
python3 -u "${SCRIPT_DIR}/train.py" \
    --seq_encoder_type transformer \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --user_dense_projector_type grouped \
    --use_time_context 0 \
    --emb_skip_threshold 5000000 \
    --batch_size 128 \
    --num_workers 8 \
    --seq_max_lens "seq_a:128,seq_b:128,seq_c:256,seq_d:256" \
    --num_epochs 20 \
    --sparse_lr 0.05 \
    --dropout_rate 0.01 \
    --patience 15 \
    --reinit_sparse_after_epoch 0 \
    --reinit_cardinality_threshold 0 \
    --loss_type bce \
    --amp \
    --amp_dtype bfloat16 \
    "$@"
```

---

## 8. 下一轮最推荐实验：D02_grouped_dense_swiglu

实验目的：

```text
验证 grouped user_dense 是否能和历史强序列编码器 swiglu 叠加。
```

D02 相对 D01 只允许改一个变量：

```text
seq_encoder_type: transformer -> swiglu
```

D02 配置：

```bash
python3 -u "${SCRIPT_DIR}/train.py" \
    --seq_encoder_type swiglu \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --user_dense_projector_type grouped \
    --use_time_context 0 \
    --emb_skip_threshold 5000000 \
    --batch_size 128 \
    --num_workers 8 \
    --seq_max_lens "seq_a:128,seq_b:128,seq_c:256,seq_d:256" \
    --num_epochs 20 \
    --sparse_lr 0.05 \
    --dropout_rate 0.01 \
    --patience 15 \
    --reinit_sparse_after_epoch 0 \
    --reinit_cardinality_threshold 0 \
    --loss_type bce \
    --amp \
    --amp_dtype bfloat16 \
    "$@"
```

判定规则：

```text
D02 > 0.818095:
  D02 成为新 best，后续以 swiglu + grouped dense 为主线。

0.816 <= D02 <= 0.818095:
  D01 transformer + grouped dense 仍为主线，D02 记为近似持平。

D02 < 0.816:
  说明 grouped dense 当前和 transformer 更适配，回到 D01。
```

是否需要同步 eval：

```text
不需要新增结构代码，因为 swiglu/transformer 都是已有 seq_encoder_type。
但必须确保 train_config.json 写入 seq_encoder_type，eval/infer.py 从 checkpoint 加载该配置。
```

---

## 9. D02 之后的候选

### 9.1 H01_clean_hash_on_D01

只有在 D02 结果出来后再做。

目的：

```text
验证 hash embedding 是否能和 D01 grouped dense 叠加。
```

要求：

```text
以 D01 为 baseline；
只开启 hash embedding；
不改 seq_encoder_type；
不改 time_context；
不改 sparse_lr；
不改 dropout/loss/seq_len；
先做 seq_b 或 seq_c 单字段/单域 ablation；
不要一次性 all hash。
```

### 9.2 D01_repeat_or_seed_check

由于 D01 一次性从 0.808496 提到 0.818095，幅度很大，建议做稳定性确认：

```text
D01 same config rerun
或 seed=2026 / seed=3407
或 best epoch checkpoint 对比
```

不过如果 eval 配额紧张，优先 D02。

### 9.3 D01_checkpoint_early_epoch_probe

D01 valid 从 epoch6 后下降，best epoch=6。可以确认 official 使用的是 AUC best checkpoint，而不是 last。

必须检查：

```text
checkpoint_select_metric=auc
best_model/model.pt 对应 epoch6
train_config.json 完整
schema.json 完整
```

---

## 10. 当前禁试清单

明确禁止下一轮提交：

```text
D01 + T01
D01 + sparse_lr=0.08
D01 + dense warmup v1
D01 + DIN v1
D01 + old UE split v1
D01 + focal/loss/dropout/参数量混合包
D01 + time_window
D01 + use_seq_time_delta_proj
D01 + seq*2
D01 + 7M threshold
```

原因：

这些方向已经在 official eval 或队友交接文档中证明低收益/负收益，不能再消耗 eval 配额。

---

## 11. 给 Codex 的立即任务

Codex 下一步不要直接开发新模型，先执行以下任务。

### 阶段 1：确认日志和配置

请 Codex 从日志或 checkpoint 中确认：

```text
E00 train_config.json:
  seq_encoder_type=transformer
  user_dense_projector_type=flat
  use_time_context=0
  official eval=0.808496

T01 train_config.json:
  seq_encoder_type=transformer
  user_dense_projector_type=flat
  use_time_context=1
  official eval=0.801715

D01 train_config.json:
  seq_encoder_type=transformer
  user_dense_projector_type=grouped
  use_time_context=0
  official eval=0.818095
```

### 阶段 2：更新文档

更新：

```text
docs/EXPERIMENT_TRACKING.md
docs/3-实验记录/当前实验交接.md
docs/1-核心蓝图/推荐算法优化蓝图.md
```

### 阶段 3：修复 run.sh

把 active block 改成当前 D01 新 best，并显式写：

```bash
--seq_encoder_type transformer
--user_dense_projector_type grouped
--use_time_context 0
```

同时准备 D02 注释块。

### 阶段 4：准备 D02

生成 `D02_grouped_dense_swiglu` 的 run.sh/script_args，不要混入任何其他变量。

### 阶段 5：输出给我确认

Codex 必须回复：

```text
1. 已确认三次实验的 train_config 关键字段；
2. 已更新哪些文档；
3. run.sh 当前 active block 是什么；
4. D02 的完整命令块；
5. 是否需要同步 eval；
6. 当前禁试清单；
7. 下一次 official eval 推荐是否为 D02。
```

---

## 12. 一句话总结

当前最重要变化：

```text
D01_grouped_user_dense_single_token = 0.818095，成为新 best。
这证明 user_dense 分组是核心提分点。
下一步最干净、最有希望的实验是：
D02 = D01 + --seq_encoder_type swiglu。
```
