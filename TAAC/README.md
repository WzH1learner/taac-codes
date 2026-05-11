# TAAC2026 PCVRHyFormer 项目入口

本项目围绕 TAAC2026 统一推荐挑战赛的 pCVR 预估任务展开，核心目标是在官方 eval AUC 上稳定超过当前最佳基线，并把每一次实验沉淀为可复用的知识资产。

新成员请从 [docs/INDEX.md](docs/INDEX.md) 开始阅读。核心优化路线、当前最佳配置、已否定实验和后续优先级，都以 [推荐算法优化蓝图](docs/1-核心蓝图/推荐算法优化蓝图.md) 为准。

## 快速导航

| 入口 | 用途 |
|---|---|
| [docs/INDEX.md](docs/INDEX.md) | 文档总入口与目录说明 |
| [推荐算法优化蓝图](docs/1-核心蓝图/推荐算法优化蓝图.md) | 当前唯一可信的优化路线图 |
| [当前实验交接](docs/3-实验记录/当前实验交接.md) | 当前最佳基线、待办和禁试路线 |
| [平台规则与提交规范](docs/2-参考资料/比赛平台规则与提交规范.md) | 训练、发布、eval 的平台约束 |
| [样例数据说明](docs/2-参考资料/样例数据说明.md) | 数据字段结构与 label 定义 |

## 当前最佳基线

```text
official eval AUC = 0.809921
seq_encoder_type = swiglu
ns_tokenizer_type = rankmixer
emb_skip_threshold = 5000000
seq_max_lens = seq_a:128,seq_b:128,seq_c:256,seq_d:256
loss_type = bce
sparse_lr = 0.05
reinit_sparse_after_epoch = 0
reinit_cardinality_threshold = 0
```

## 代码目录

```text
train/       训练入口、数据管道、模型与 trainer
eval/        官方推理入口与 eval 侧兼容代码
sample_data/ 样例数据说明与本地调试数据
docs/        活跃知识库与历史归档
research/    保留代码型研究工具，例如 inspect_pcvr_structure.py
```

结构性模型或输入字段改动必须同步 `train/` 与 `eval/`。只改优化器、loss、valid split 或 checkpoint 策略时，通常只需要更新训练侧。
