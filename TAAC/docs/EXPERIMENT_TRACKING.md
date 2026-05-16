# EXPERIMENT TRACKING

Official eval AUC is the decision metric. Validation AUC, LogLoss, Brier, and
prob_mean/label_mean are candidate filters only.

## Metric Review Principle - 2026-05-16

Near the preliminary deadline, the goal is not just to run more ideas; it is to
turn each run into sharper decisions that can improve official eval AUC. Official
eval AUC remains the final scoreboard, but training logs are useful diagnostics.
Do not judge an experiment by valid AUC alone, and do not discard every failed
official run without reading what its metrics taught us.

How to read the main validation metrics:

| Metric | What it tells us | How to use it |
| --- | --- | --- |
| `valid AUC` | Ranking/separation quality on the local split. | Main filter for whether a checkpoint deserves official eval. |
| `LogLoss` | Probability quality and confidence penalty. | If AUC is near D01 but LogLoss is clearly better, keep the direction as a candidate. |
| `Brier` | Calibration and squared probability error. | Useful together with LogLoss; lower means probabilities are less distorted. |
| `prob_mean / label_mean` | Base-rate calibration. | `prob_mean` far below label rate means conservative underprediction; far above means overprediction. |
| `prob_std` and `logit_std` | Output spread/confidence. | Higher can mean better separation inside one run, but it is not universally better across different model families. |

Important rule for `prob_std`: within D01, `prob_std` rose together with valid
AUC, so it behaved like a separation signal. Across other experiments this did
not hold. High `prob_std` with worse LogLoss can mean confidently wrong scores;
low `prob_std` with low `prob_mean` can mean the model is too conservative.

Candidate decision rule:

```text
1. Official eval AUC decides the leaderboard.
2. Valid AUC decides whether an experiment deserves official eval budget.
3. LogLoss/Brier/prob_mean decide whether a near-tie is worth keeping.
4. Failed official experiments still teach what to reject or redesign.
5. For time/pair/history ideas, run EDA first, then train a single-variable model.
```

Metric examples from recent runs:

| Run | Valid AUC | LogLoss | Brier | prob_mean / label_mean | prob_std | Diagnostic read |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| D01 current best | `0.864399` | `0.224305` | `0.064283` | `0.093914 / 0.096785` | `0.164718` | Best verified ranking; calibration acceptable. |
| emb_skip 5w/2w offline candidate | `0.864382` | `0.222173` | `0.063951` | `0.098231 / 0.096785` | `0.155530` | Near-tie AUC with better calibration; do not discard solely because AUC is tiny lower. |
| P2 current attempt | `0.863351` | `0.223793` | `0.064284` | `0.087573 / 0.096785` | `0.146727` | Too conservative; current P2 residual is not strong, but pair/time EDA signal remains worth redesigning. |
| G01 group tokenizer | `0.861724` | `0.225285` | `0.064594` | `0.083881 / 0.096785` | `0.143048` | Lower AUC and underprediction; not a mainline without redesign. |

Time-feature lesson: T01 and R01 failed official eval, so root/global time and
global recency are rejected as implemented. This does not mean all time signals
are useless. Next time-related attempts must start from EDA and focus on
target-conditioned history signals, such as matched recency, per-domain gap
lift, and train-tail stability.

Time signal EDA command:

```bash
python research/code/time_signal_eda.py \
  --data_path "${TRAIN_DATA_PATH:-/data_ams/academic_training_data}" \
  --max_files 10 \
  --max_rows 50000 \
  --split_by_valid_tail 1 \
  --output "${TRAIN_CKPT_PATH:-research/reports}/time_signal_eda.md"
```

Smoke first if needed:

```bash
python research/code/time_signal_eda.py \
  --data_path "${TRAIN_DATA_PATH:-/data_ams/academic_training_data}" \
  --max_files 3 \
  --max_rows 10000 \
  --split_by_valid_tail 1 \
  --output "${TRAIN_CKPT_PATH:-research/reports}/time_signal_eda_smoke.md"
```

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
| `A01_aligned_user_int_dense_weighted_pooling_no_compile` | Next main experiment | Uses README same-fid alignment between `user_int_feats_{fid}` and `user_dense_feats_{fid}` as a small final residual. Default off; no compile. |
| `P2_pair_time_match` | Paused candidate | Pair EDA shows target-history exact-match signal, but current priority is A01. Do not continue P2 v1/v2 until A01 is closed. |
| Global recency / time_context | Rejected | T01 and R01 both failed official eval. Do not continue root timestamp, weekday/hour bucket, or global recency stats. |
| SwiGLU route | Downgraded | Clean D02 official `0.814760` is below D01 `0.818095`. |

## A01_aligned_user_int_dense_weighted_pooling_no_compile

Code status: implemented behind a default-off switch. Current D01 active path
keeps `use_aligned_user_int_dense=0`.

Only variable vs D01:

```text
--use_aligned_user_int_dense 1
--aligned_user_int_dense_gate_init 0.05
--aligned_user_int_dense_fids_json ""
```

Design:

```text
candidate fids = 62, 63, 64, 65, 66, 89, 90, 91
reuse existing user_int embedding tables
same-fid user_dense values become weights for valid user_int ids
pooled aligned vectors -> small MLP -> final residual
final_repr = final_repr + aligned_gate * aligned_repr
```

Guardrails:

```text
No RankMixer token is added.
Do not enable time_context, seq_recent_stats, or pair_dense.
Do not change loss/lr/dropout/seq_len/emb_skip_threshold.
Run no-compile only; F00b reduce-overhead compile is rejected.
```

## F00_fast_screening

Purpose: reduce experiment feedback time. F00 is for screening and timing, not
for declaring final official results.

| Run | Command | Use |
| --- | --- | --- |
| `D01_no_compile_timing` | `bash TAAC/train/run.sh` | Current D01 unchanged, with epoch timing logs. |
| `F00b_D01_compile_baseline` | `bash TAAC/train/run.sh --torch_compile --compile_mode reduce-overhead` | Rejected: crashed on cloud before epoch 1 completed. |

Failure signature:

```text
RuntimeError: Expected curr_block->next == nullptr to be true, but got false
location: torch/_inductor/cudagraph_trees.py / _cuda_setCheckpointPoolState
```

Decision:

```text
Do not use --torch_compile --compile_mode reduce-overhead on the cloud runtime.
There is no valid D01_compile_baseline.
Do not compare future experiments against compile numbers.
Continue with no-compile D01 timing as the only valid F00 record.
Default run.sh behavior remains D01 no-compile.
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
