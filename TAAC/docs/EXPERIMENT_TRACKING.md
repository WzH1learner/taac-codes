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
| emb_skip 2w | `0.864158` | `0.222410` | `0.064064` | `0.098332 / 0.096785` | `0.156216` | Better calibration did not transfer; official `0.808294`, reject. |
| P2 current attempt | `0.863351` | `0.223793` | `0.064284` | `0.087573 / 0.096785` | `0.146727` | Too conservative; current P2 residual is not strong, but pair/time EDA signal remains worth redesigning. |
| G01 group tokenizer | `0.861724` | `0.225285` | `0.064594` | `0.083881 / 0.096785` | `0.143048` | Lower AUC and underprediction; not a mainline without redesign. |

Time-feature lesson: T01 and R01 failed official eval, so root/global time and
global recency are rejected as implemented. This does not mean all time signals
are useless. Next time-related attempts must start from EDA and focus on
target-conditioned history signals, such as matched recency, per-domain gap
lift, and train-tail stability.

Time signal EDA command:

```bash
mkdir -p "${TRAIN_CKPT_PATH:-research/reports}"
python3 -u research/code/time_signal_eda.py \
  --data_path "${TRAIN_DATA_PATH:-/data_ams/academic_training_data}" \
  --max_files 1000 \
  --max_rows 200000 \
  --recursive_files 1 \
  --write_csv 1 \
  --summary_top_k 30 \
  --split_by_valid_tail 1 \
  --output "${TRAIN_CKPT_PATH:-research/reports}/time_signal_eda.md" \
  2>&1 | tee "${TRAIN_CKPT_PATH:-research/reports}/time_signal_eda.log"
```

Cloud-log friendly printout:

```bash
REPORT="${TRAIN_CKPT_PATH:-research/reports}/time_signal_eda.md"
awk '/^## Compact Summary/{flag=1} /^## Global Domain Recency Lift/{flag=0} flag{print}' "$REPORT"
awk '/^## Target-Matched Recency Lift/{flag=1; n=0} /^## Target-Matched Recency Stability/{flag=0} flag && n++ < 65{print}' "$REPORT"
awk '/^## Target-Matched Recency Stability/{flag=1; n=0} /^## How To Use This Report/{flag=0} flag && n++ < 45{print}' "$REPORT"
```

Cloud summary helper:

```bash
python3 -u research/code/print_eda_report_summary.py \
  --out_dir "${TRAIN_CKPT_PATH}" \
  --top_k 120
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
| D01_emb_skip_2w | D01 with lower embedding skip threshold | 6 | `0.864158` | `0.222410` | `0.808294` | Rejected by official eval. Do not continue this line. |
| P4_seq_d_target_match_token_flag_full_5flags | D01 + token-level seq_d target-match flags `[12,13,9,83,6]`, gate `0.01` | TBD | TBD | TBD | `0.808359` | Rejected. Infer confirmed P4 was enabled; do not continue 5-flag version. |
| P4b_seq_d_target_match_token_flag_lite_83_6_gate005 | D01 + token-level seq_d target-match flags `[83,6]`, gate `0.005` | TBD | TBD | TBD | `0.8028` | Rejected. Pause P4 series; do not continue P4c / flag-specific gates. |
| S01_seq_d_sideinfo_grouped_projector_25_24 | D01 + grouped seq_d sideinfo projector for fids `[25,24]` | 6 | `0.864147` | `0.222898` | `0.801289` | Rejected. Official score is far below D01. |
| D03_seq_len_192_384 | D01 + longer sequence length | TBD | `0.864719` | TBD | `0.811244` | Rejected. Valid improved but official did not transfer. |
| A02_user_dense_int_pair_gate_62_66_no8991 | D01 + gated same-fid user_int/user_dense side branch, fids `[62..66]`, exclude `[89,90,91]`, gate `0.01` | 8 | `0.863886` | `0.223215` | TBD | Weak candidate only. AUC below D01 but still rising; prob_mean normal. Do not eval unless AUC >= `0.86435` with normal prob_mean. |

## Active Direction

| Direction | Status | Rationale |
| --- | --- | --- |
| `D01 low-risk rerun / seed sweep` | Next priority | Official test is a very narrow time window and valid ranking has repeatedly disagreed with official. Return near D01 and search low-risk knobs. |
| `checkpoint top-k infrastructure` | Required before more reruns | D01 reruns / seed sweep must keep top-k checkpoints and compare AUC near-ties by LogLoss/Brier/prob-gap. |
| `I01/P3/P4/S01/A01/P2/D03` | Paused | Do not continue new structures until D01-adjacent search is exhausted. |
| `P4 target-match token flags` | Paused / rejected | P4 full official `0.808359`; P4b official `0.8028`. Do not continue P4c or flag-specific gates for now. |
| `A02_user_dense_int_pair_gate` | Weak candidate / no eval yet | epoch8 valid AUC `0.863886`, LogLoss `0.223215`, Brier `0.064170`, prob_mean `0.097750`. Can test longer convergence or gate `0.02`; keep `[89,90,91]` excluded and do not mix with T02. |
| `test_aware_feature_audit` | Completed for P4 full | `rows_scanned=200000`, `parquet_files_scanned=1000`, `strong_and_stable_specs=3`, `risky_specs=2`. |
| `P3a_target_matched_recency_any_lite` | Rejected as residual route | Valid AUC `0.864184`, LogLoss `0.225655`, prob_mean `0.080864`; do not continue final_repr residual recency. |
| `A01_aligned_user_int_dense_weighted_pooling_no_compile` | Rejected as residual route | prob_mean was compressed; do not continue A01 before stronger evidence. |
| `P2_pair_time_match` | Paused candidate | Pair EDA shows target-history exact-match signal, but all-pair final residual underperformed. Current priority is P4 token-level injection. |
| Global recency / time_context | Rejected | T01 and R01 both failed official eval. Do not continue root timestamp, weekday/hour bucket, or global recency stats. |
| SwiGLU route | Downgraded | Clean D02 official `0.814760` is below D01 `0.818095`. |

## P4 Target-Match Token Flags

P4 full 5-flag result:

```text
P4_seq_d_target_match_token_flag_full_5flags
official eval AUC = 0.808359
decision = reject
```

Infer confirmed the intended path was active:

```text
seq_target_match_flags enabled: domain=seq_d, num_flags=5, gate_init=0.0100
target_matched_recency disabled
pair_dense disabled
seq_recent_stats disabled
aligned_user_int_dense disabled
seq_target_match_flags configured: dims_by_domain={'seq_d': 5}, active_specs=5
```

Test-aware EDA:

```text
rows_scanned = 200000
parquet_files_scanned = 1000
strong_and_stable_specs = 3
risky_specs = 2
stable specs = 13|seq_d|25, 83|seq_d|25, 6|seq_d|24
risky specs = 12|seq_d|25, 9|seq_d|25
```

Why full P4 failed / what to avoid:

```text
12 reverses in official-like tails: tail_2h_lift=0.480613, tail_6h_lift=0.716015.
9 is not stable enough: tail_2h_lift≈0.999.
13 is stable but true_rate > 0.83 with high matched token count; direct token-level flag may perturb too many seq_d tokens.
Do not continue the 5-flag version.
```

Next lite candidate:

```text
P4b_seq_d_target_match_token_flag_lite_83_6_gate005
```

Only variables vs D01:

```text
--use_seq_target_match_flags 1
--seq_target_match_flag_gate_init 0.005
--seq_target_match_flag_specs_json '[{"item_fid":83,"domain":"seq_d","side_fid":25},{"item_fid":6,"domain":"seq_d","side_fid":24}]'
```

Training command:

```bash
bash TAAC/train/run.sh \
  --use_seq_target_match_flags 1 \
  --seq_target_match_flag_gate_init 0.005 \
  --seq_target_match_flag_specs_json '[{"item_fid":83,"domain":"seq_d","side_fid":25},{"item_fid":6,"domain":"seq_d","side_fid":24}]'
```

Design: dataset emits `seq_target_match_flags["seq_d"]` with shape
`[B, L, num_flags]`. The model applies `Linear(num_flags, d_model)` and adds
`gate * flag_emb` to `seq_d` token embeddings before the sequence encoder.
No final_repr residual, no RankMixer token, no recency window/count feature.

Do not enable P3a / pair_dense / A01 / seq_recent_stats / time_context /
compile. Use direct `TAAC/train/run.sh`; avoid nested wrapper commands because
platform logs may hide the real active args.

## Checkpoint Selection Infrastructure

Defaults preserve D01 behavior:

```text
--keep_top_k_checkpoints 1
--checkpoint_select_metric auc
--always_save_last_checkpoint 0
```

For D01 reruns / seed sweep, use top-k:

```text
--keep_top_k_checkpoints 3
--checkpoint_select_metric auc_then_logloss
```

`auc_then_logloss` ranking:

```text
1. Higher valid AUC wins.
2. If AUC difference <= 0.0003, lower LogLoss wins.
3. Then lower Brier.
4. Then smaller abs(prob_mean - label_mean).
```

Checkpoint directory names include `step`, `epoch`, `AUC`, `LogLoss`, and
`Brier` for platform-side selection. `--always_save_last_checkpoint 1` keeps a
rolling `.last` checkpoint in addition to top-k/best.

Low-risk candidates:

```bash
bash TAAC/train/run.sh --seed 2026 --num_epochs 10 --patience 5
bash TAAC/train/run.sh --dropout_rate 0.0 --num_epochs 10 --patience 5
bash TAAC/train/run.sh --sparse_lr 0.03 --num_epochs 10 --patience 5
```

Optional top-k add-on:

```bash
--keep_top_k_checkpoints 3 --checkpoint_select_metric auc_then_logloss
```

Archived full-P4 command:

```bash
bash TAAC/train/run.sh \
  --use_seq_target_match_flags 1 \
  --seq_target_match_flag_gate_init 0.01 \
  --seq_target_match_flag_specs_json ""
```

P4b lite result:

```text
P4b_seq_d_target_match_token_flag_lite_83_6_gate005
official eval AUC = 0.8028
decision = reject
```

Conclusion: pause the P4 series. Do not continue P4c or flag-specific gates
until there is new EDA evidence.

## S01_seq_d_sideinfo_grouped_projector_25_24

Result:

```text
best epoch = 6
valid AUC = 0.8641473902303051
LogLoss = 0.22289758920669556
Brier = 0.0640855
label_mean = 0.0967852
prob_mean = 0.0967916
prob_std = 0.158664
official eval = CUDA OOM, no score
```

Infer confirmed S01 was enabled and other experimental branches were off:

```text
seq_d grouped side projector enabled: important_fids=[25,24], important_positions=[7,8], other_count=7
seq_target_match_flags disabled
target_matched_recency disabled
pair_dense disabled
seq_recent_stats disabled
```

Conclusion: S01 is a weak candidate. It improves calibration metrics versus
D01 but has lower valid AUC; official eval failed by CUDA OOM, not strict load
or config error. Retry only if eval quota is loose.

Audit command:

```bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EDA_SCRIPT="${SCRIPT_DIR}/test_aware_feature_audit.py"
if [ ! -f "${EDA_SCRIPT}" ]; then EDA_SCRIPT="${SCRIPT_DIR}/TAAC/research/code/test_aware_feature_audit.py"; fi
if [ ! -f "${EDA_SCRIPT}" ]; then EDA_SCRIPT="${SCRIPT_DIR}/research/code/test_aware_feature_audit.py"; fi
python3 -u "${EDA_SCRIPT}" \
  --data_path "${TRAIN_DATA_PATH:-/data_ams/academic_training_data}" \
  --max_files 1000 \
  --max_rows 200000 \
  --recursive_files 1 \
  --write_csv 1 \
  --summary_top_k 50 \
  --output "${TRAIN_CKPT_PATH:-research/reports}/test_aware_feature_audit.md"
```

Log-friendly report preview:

```bash
awk '/^## Compact Summary/{flag=1} /^## Spec Details/{flag=0} flag{print}' "${TRAIN_CKPT_PATH}/test_aware_feature_audit.md"
head -80 "${TRAIN_CKPT_PATH}/test_aware_feature_audit_specs.csv"
head -80 "${TRAIN_CKPT_PATH}/test_aware_feature_audit_num_flags.csv"
```

## P3a_target_matched_recency_any_lite

Only variable vs D01:

```text
--use_target_matched_recency 1
--target_matched_recency_gate_init 0.005
--target_matched_recency_feature_mode any_only
--target_matched_recency_pairs_windows_json ""
```

Default P3a features:

```text
[12, seq_d, 25] windows = 30m,2h,6h,1d,3d,7d,30d
[6,  seq_d, 24] windows = 30m,2h,6h,1d,3d,7d,30d
[13, seq_d, 25] windows = 1d,3d,7d,30d
[83, seq_d, 25] windows = 30m,2h,6h,1d,3d,7d,30d
[9,  seq_d, 25] windows = 30m,2h,6h,1d,3d,7d,30d
```

Feature count: `32`. First version is `recent_any` only. Do not add `5`/`81`,
log_count, last_gap, bare match_any, or global recency.

Platform command:

```bash
bash TAAC/train/run.sh \
  --use_target_matched_recency 1 \
  --target_matched_recency_gate_init 0.005 \
  --target_matched_recency_feature_mode any_only \
  --target_matched_recency_pairs_windows_json ""
```

Expected startup log:

```text
use_target_matched_recency=1
target_matched_recency_dim=32
target_matched_recency_gate_init=0.005
target_matched_recency_feature_mode=any_only
target_matched_recency_feats diagnostics: dim=32
seq_encoder_type=transformer
user_dense_projector_type=grouped
use_time_context=0
use_seq_recent_stats=0
use_pair_dense=0
T=16
```

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

## A02_user_dense_int_pair_gate_62_66_no8991

Current result:

```text
best observed epoch = 8
valid AUC = 0.8638862957897601
LogLoss = 0.22321513295173645
Brier = 0.0641699
label_mean = 0.0967852
prob_mean = 0.0977497
prob_std = 0.162892
```

Interpretation:

```text
AUC is below D01, so do not eval this checkpoint.
Within-run valid AUC was still rising by epoch8.
prob_mean is healthy and not compressed like P3/P4/A01.
LogLoss/Brier are reasonable.
This looks more like slow convergence or a weak side branch than a broken route.
```

Allowed follow-ups:

```bash
# A02_long_user_dense_int_pair_gate_epoch14
bash TAAC/train/run.sh \
  --use_user_dense_int_pair_gate 1 \
  --user_dense_int_pair_gate_init 0.01 \
  --user_dense_int_pair_fids_json '[62,63,64,65,66]' \
  --user_dense_int_pair_exclude_fids_json '[89,90,91]' \
  --num_epochs 14 \
  --patience 6 \
  --keep_top_k_checkpoints 3 \
  --checkpoint_select_metric auc_then_logloss
```

```bash
# A02_gate02_user_dense_int_pair_gate_62_66_no8991
bash TAAC/train/run.sh \
  --use_user_dense_int_pair_gate 1 \
  --user_dense_int_pair_gate_init 0.02 \
  --user_dense_int_pair_fids_json '[62,63,64,65,66]' \
  --user_dense_int_pair_exclude_fids_json '[89,90,91]' \
  --num_epochs 10 \
  --patience 5 \
  --keep_top_k_checkpoints 3 \
  --checkpoint_select_metric auc_then_logloss
```

Guardrails:

```text
Do not add back 89/90/91.
Do not mix with T02.
Do not change sparse_lr/dropout/seq_len.
Eval threshold: valid AUC >= 0.86435 and prob_mean normal.
If LogLoss/Brier improve but AUC stays below threshold, do not eval.
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
mkdir -p "${TRAIN_CKPT_PATH:-research/reports}"
python3 -u research/code/pair_match_eda.py \
  --data_path "${TRAIN_DATA_PATH:-/data_ams/academic_training_data}" \
  --max_files 1000 \
  --max_rows 200000 \
  --recursive_files 1 \
  --write_csv 1 \
  --summary_top_k 30 \
  --min_match_count 50 \
  --min_recent_match_count 10 \
  --top_k 50 \
  --focus_domain seq_d \
  --focus_side_fid 25 \
  --split_by_valid_tail 1 \
  --output "${TRAIN_CKPT_PATH:-research/reports}/pair_match_eda_v2.md" \
  2>&1 | tee "${TRAIN_CKPT_PATH:-research/reports}/pair_match_eda_v2.log"
```

Cloud-log friendly printout:

```bash
REPORT="${TRAIN_CKPT_PATH:-research/reports}/pair_match_eda_v2.md"
awk '/^## Compact Summary/{flag=1} /^## Schema Candidates/{flag=0} flag{print}' "$REPORT"
awk '/^## Stable Candidate Pairs/{flag=1; n=0} /^## Unstable High-Lift/{flag=0} flag && n++ < 45{print}' "$REPORT"
awk '/^## Focus Table/{flag=1; n=0} /^## Top Pairs By Lift/{flag=0} flag && n++ < 45{print}' "$REPORT"
awk '/^## First 90% vs Tail 10% Stability/{flag=1; n=0} /^## Recommendations/{flag=0} flag && n++ < 45{print}' "$REPORT"
```

CSV outputs:

```text
pair_match_all.csv
pair_match_stable.csv
pair_match_focus.csv
pair_match_stability.csv
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
