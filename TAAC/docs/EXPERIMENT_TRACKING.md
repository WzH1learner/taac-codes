# EXPERIMENT TRACKING

Official eval AUC is the decision metric. Validation AUC, LogLoss, Brier, and
prob_mean/label_mean are candidate filters only.

## Current Best

| Field | Value |
| --- | --- |
| Current best | `D01_grouped_user_dense_single_token` |
| Official eval AUC | `0.818095` |
| Best valid AUC | `0.864399@epoch6` |
| Valid LogLoss at best epoch | `0.224305` |
| Valid prob_mean / label_mean | `0.093914 / 0.096785` |
| Mainline | `transformer + rankmixer + 5M + BCE + short seq + grouped user_dense + no time_context + no seq_recent_stats` |
| Previous fallback | `0.809921`, historical `swiglu + rankmixer + 5M + BCE + short seq` |

## Confirmed Experiments

| ID | Config | Best Epoch | Best Valid AUC | Valid LogLoss | Official Eval AUC | Conclusion |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| E00_reproduce_080992 | `transformer`, `rankmixer`, `emb_skip_threshold=5M`, `user_dense_projector_type=flat`, `use_time_context=0`, BCE, short seq | 5 | `0.860380` | `0.226409` | `0.808496` | Transformer + flat dense control. |
| T01_time_context | E00 + `use_time_context=1` | 4 | `0.860832` | `0.225411` | `0.801715` | Rejected. Root/global time context has strong official drift risk. |
| D01_grouped_user_dense_single_token | E00 + `user_dense_projector_type=grouped` | 6 | `0.864399` | `0.224305` | `0.818095` | Current best. Grouped user_dense is the strongest verified change. |
| D02_grouped_dense_swiglu_clean | D01 + `seq_encoder_type=swiglu` | TBD | `0.864255` | `0.223297` | `0.814760` | Rejected as mainline. SwiGLU + grouped dense underperforms D01 official. |
| R01_seq_recent_stats_residual_v1 | D01 + `use_seq_recent_stats=1`, gate `0.1` | TBD | TBD | TBD | `0.808144` | Rejected. Global recency stats do not transfer to official eval. |
| G01_group_tokenizer | Group tokenizer candidate | TBD | `0.8617` | TBD | TBD | Keep as possible G02 candidate, but not the mainline. |

## Active Direction

| Direction | Status | Rationale |
| --- | --- | --- |
| `P2_pair_time_match` | Next feature candidate | Pair EDA shows stable target-history exact-match signal around `seq_d` side_fid `25`, especially item_fid `13`. This is not global recency; it is target-specific history match plus recency. |
| Global recency / time_context | Rejected | T01 and R01 both failed official eval. Do not continue root timestamp, weekday/hour bucket, or global recency stats. |
| SwiGLU route | Downgraded | Clean D02 official `0.814760` is below D01 `0.818095`. |

## F00_fast_screening

Purpose: reduce experiment feedback time. F00 is for screening and timing, not
for declaring final official results.

| Run | Command | Use |
| --- | --- | --- |
| `D01_no_compile_timing` | `bash TAAC/train/run.sh` | Current D01 unchanged, with epoch timing logs. |
| `D01_compile_baseline` | `bash TAAC/train/run.sh --torch_compile --compile_mode reduce-overhead` | Compile baseline for later compile-only screening comparisons. |
| Compile smoke | `bash TAAC/train/run.sh --torch_compile --compile_mode reduce-overhead --num_epochs 2 --train_ratio 0.2` | Only verifies that compile path runs; not an AUC decision. |

Rules:

```text
compile=True is only for direction screening.
All compile experiments must be compared with D01_compile_baseline.
Compile results must not be directly compared with no-compile D01 official best.
If a direction works under compile, retrain it no-compile before official eval.
Official eval should prefer no-compile checkpoints unless explicitly marked compile.
Default run.sh behavior stays D01 no-compile.
```

## Pair EDA Findings

Small EDA run:

```text
rows_scanned = 10313
```

Important stable-looking candidates:

| item_fid | domain | side_fid | match_any_rate | recent_match_any_2h_rate | positive_rate_when_match | positive_rate_when_no_match | lift |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 13 | `seq_d` | 25 | `0.828081` | `0.208863` | `0.099415` | `0.050761` | `1.958466` |
| 81 | `seq_d` | 25 | TBD | TBD | TBD | TBD | `1.189039` |
| 9 | `seq_d` | 25 | TBD | TBD | TBD | TBD | `1.432972` |
| 5 | `seq_d` | 25 | TBD | TBD | TBD | TBD | `1.309826` |
| 83 | `seq_d` | 25 | TBD | TBD | TBD | TBD | `1.318245` |
| 10 | `seq_d` | 25 | TBD | TBD | TBD | TBD | `1.148435` |
| 6 | `seq_d` | 24 | TBD | TBD | TBD | TBD | `2.290049` |

Note: many top-lift pairs have tiny match counts. P2 v1 must use only pairs with both coverage and lift.

## Pair EDA v2 Command

```bash
python research/code/pair_match_eda.py \
  --data_path "${TRAIN_DATA_PATH:-/data_ams/academic_training_data}" \
  --max_files 10 \
  --max_rows 50000 \
  --min_match_count 50 \
  --min_recent_match_count 10 \
  --top_k 50 \
  --focus_domain seq_d \
  --focus_side_fid 25 \
  --split_by_valid_tail 1 \
  --output "${TRAIN_CKPT_PATH}/pair_match_eda_v2.md"
```

## P2_pair_dense_v1 Draft

Code status: prepared behind a default-off switch. Current D01 active path keeps
`use_pair_dense=0`.

Selected pairs:

```json
[
  [13, "seq_d", 25],
  [81, "seq_d", 25],
  [9, "seq_d", 25],
  [5, "seq_d", 25],
  [83, "seq_d", 25],
  [10, "seq_d", 25],
  [6, "seq_d", 24]
]
```

For each pair, build 7 features:

```text
match_any
log1p(match_count)
recent_any_2h
log1p(recent_count_2h)
recent_any_1d
log1p(recent_count_1d)
last_gap_log = log1p(root_ts - latest_matched_ts), 0 if no match
```

Total feature dimension: `7 pairs * 7 = 49`.

Model connection:

```text
pair_dense_feats -> LayerNorm -> MLP(49 -> d_model)
final_repr = final_repr + pair_dense_gate * pair_dense_emb
pair_dense_gate_init = 0.05
```

Default must remain:

```text
use_pair_dense = 0
```

## Checkpoint Selection Plan

Future plan only, not mixed with P2:

```text
--keep_top_k_checkpoints 3
--checkpoint_select_metric auc_then_logloss
```

Selection rule:

```text
best_auc = max(valid_auc)
candidates = valid_auc >= best_auc - 0.0003
choose lower LogLoss
then lower Brier
then prob_mean closer to label_mean
```

Default remains `checkpoint_select_metric=auc` to protect D01 behavior.

## Do Not Spend Training Or Official Eval On

| Direction | Reason |
| --- | --- |
| `D01 + T01`, root timestamp, weekday/hour bucket, or global recency | T01 official `0.801715`; R01 official `0.808144`. |
| `D01 + swiglu` | Clean D02 official `0.814760`, below D01. |
| `D01 + sparse_lr=0.08` | Prior official eval `0.805707`; do not combine with D01. |
| `D01 + dense warmup v1` | Prior official eval `0.807503`; not next priority. |
| `D01 + DIN v1` | Prior official eval `0.807927`; use P2 EDA/features first. |
| `D01 + old UE split v1` | Old UE split official eval `0.808142`; D01 grouped projector is better. |
| `focal`, dropout, or parameter-count bundles | Calibration risk and unclear attribution. |
| `time_window=2h` | Official eval `0.7883`. |
| `use_seq_time_delta_proj=True` | Official eval `0.809214`, below current best. |
| `seq*2` | Official eval `0.809907`, slower and below current best. |
| `emb_skip_threshold=7M` | Official eval `0.806861`. |
| G01 group tokenizer training | Keep as G02 candidate only; not current mainline. |
