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
| Mainline | `transformer + rankmixer + 5M + BCE + short seq + grouped user_dense + no time_context` |
| Previous fallback | `0.809921`, historical `swiglu + rankmixer + 5M + BCE + short seq` |

## Confirmed Experiments

| ID | Config | Best Epoch | Best Valid AUC | Valid LogLoss | Official Eval AUC | Conclusion |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| E00_reproduce_080992 | `seq_encoder_type=transformer`, `rankmixer`, `emb_skip_threshold=5M`, `user_dense_projector_type=flat`, `use_time_context=0`, BCE, short seq | 5 | `0.860380` | `0.226409` | `0.808496` | Transformer + flat dense control. Not a strict reproduction of old `0.809921` because the earlier `run.sh` omitted `--seq_encoder_type`. |
| T01_time_context | E00 + `use_time_context=1` | 4 | `0.860832` | `0.225411` | `0.801715` | Rejected. Root timestamp/hour-style context has strong official distribution-shift risk. |
| D01_grouped_user_dense_single_token | E00 + `user_dense_projector_type=grouped` | 6 | `0.864399` | `0.224305` | `0.818095` | New best. Grouped user_dense is verified high yield. |

## In Flight

| ID | Single Variable | Full Config Delta | Decision Rule |
| --- | --- | --- | --- |
| D02_grouped_dense_swiglu | `seq_encoder_type: transformer -> swiglu` | Keep D01 unchanged except `--seq_encoder_type swiglu` | `>0.818095`: switch to swiglu + grouped dense. `0.816-0.818095`: near-tie, keep D01. `<0.816`: keep transformer + grouped dense. |

## Tonight Second Training Candidate

| ID | Single Variable | Hypothesis | Eval Rule |
| --- | --- | --- | --- |
| R01_seq_recent_stats_residual_v1 | `use_seq_recent_stats: 0 -> 1`, `seq_recent_stats_gate_init=0.1` | Relative recent activity / recency intensity may capture the useful part of time without absolute hour/weekday/root timestamp shift. | If valid AUC `>=0.864399`, consider official eval unless D02 is better. If valid AUC is close but LogLoss/Brier/prob_mean are healthier, keep as candidate. If valid drops or calibration breaks, do not eval. |

R01 keeps D01 unchanged otherwise:

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
    --use_seq_recent_stats 1 \
    --seq_recent_stats_gate_init 0.1 \
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

## Analysis Only

| ID | Script | Goal | Status |
| --- | --- | --- | --- |
| P2_pair_target_history_match_eda | `research/code/pair_match_eda.py` | Check whether current `item_int` values exactly match sequence sideinfo values and whether matches lift positives. | EDA only. Do not train P2 until the markdown report shows usable signal. |

Cloud command:

```bash
python research/code/pair_match_eda.py \
    --data_path "${TRAIN_DATA_PATH:-/data_ams/academic_training_data}" \
    --max_files 10 \
    --max_rows 50000 \
    --output research/reports/pair_match_eda.md
```

## Low-Cost Later Candidate

| ID | Seeds | Principle | Priority |
| --- | --- | --- | --- |
| S01_D01_seed_sweep | `42, 2026, 3407, 1234, 2025` | Use the best mainline after D02/R01, change only `--seed`, compare valid AUC/LogLoss/prob_mean/label_mean, and official-eval only clearly better candidates. | After D02 and R01. Seed sweep is a micro-gain check, not a structural feature. |

Notes for S01:

- CUDA kernels, SDPA, and IterableDataset ordering may still be non-deterministic, so seed results can wobble.
- Seed sweep may be worth roughly `0.001-0.002`, but it cannot replace structural improvements.
- Do not combine seed sweep with R01, hash, G01, time_context, or any other change in the same run.

## Checkpoint / Eval Notes

- R01 adds model parameters only when `use_seq_recent_stats=1`; with `0`, D01 path is state-dict compatible.
- `train_config.json` must contain `use_seq_recent_stats`, `seq_recent_stats_dim`, and `seq_recent_stats_gate_init`.
- Before official eval, confirm checkpoint has `model.pt`, `train_config.json`, and `schema.json`.
- For any future `group` tokenizer run, checkpoint must also include `ns_groups.json`.

## Do Not Spend Training Or Official Eval On

| Direction | Reason |
| --- | --- |
| `D01 + T01` or any current `time_context` combo | T01 official eval dropped to `0.801715`. |
| `D01 + sparse_lr=0.08` | Prior official eval `0.805707`; do not combine with D01 before a clean hypothesis. |
| `D01 + dense warmup v1` | Prior official eval `0.807503`; not next priority. |
| `D01 + DIN v1` | Prior official eval `0.807927`; do not train target-history attention before P2 EDA. |
| `D01 + old UE split v1` | Old UE split official eval `0.808142`; not a counterexample to D01 grouped projector. |
| `focal`, dropout, or parameter-count bundles | Calibration risk and unclear attribution. |
| `time_window=2h` | Official eval `0.7883`. |
| `use_seq_time_delta_proj=True` | Official eval `0.809214`, below current best. |
| `seq*2` | Official eval `0.809907`, slower and below current best. |
| `emb_skip_threshold=7M` | Official eval `0.806861`. |
| hash embedding training | Not tonight; keep after D02/R01 and only as clean ablation. |
| G01 group tokenizer training | Not tonight; do not mix with D01/R01. |
