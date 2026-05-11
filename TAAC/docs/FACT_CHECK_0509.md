# FACT CHECK 0509

## 0.80992 source

Confirmed from `README.md` and `docs/1-核心蓝图/推荐算法优化蓝图.md`:

- Best official eval AUC: `0.809921` (reported by user as `0.80992`).
- Backbone: `SwiGLU + RankMixer + BCE + short seq`.
- Critical config:
  - `ns_tokenizer_type=rankmixer`
  - `ns_groups_json=""`
  - `user_ns_tokens=5`
  - `item_ns_tokens=2`
  - `num_queries=2`
  - `d_model=64`
  - `seq_max_lens=seq_a:128,seq_b:128,seq_c:256,seq_d:256`
  - `emb_skip_threshold=5000000`
  - `checkpoint_select_metric=auc`
- Historical notes say best valid AUC was around `0.861063@epoch6`, official eval AUC `0.809921`, inference around `129.65s`.

Local drift found before this change:

- `train/run.sh` active block used `emb_skip_threshold=1000000`, not the documented `5000000` official-best setting.
- `train/run.sh` active block still used RankMixer and short seq, but was not an exact reproduction of the best official checkpoint.

Action taken:

- Restored `train/run.sh` active block to the 5M official-best backbone and made `--user_dense_projector_type flat`, `--use_time_context 0` explicit.

## Train / Eval Config Consistency

Checked:

- `train/train.py` writes `train_config=vars(args)` into checkpoint sidecars through `trainer.py`.
- `eval/infer.py` loads `train_config.json` first and falls back to hardcoded defaults only if missing.
- `eval/infer.py` resolves `ns_groups_json` relative to `MODEL_OUTPUT_PATH` when `trainer.py` copied it as a sidecar.

Added to train/eval structural config:

- `user_dense_projector_type`
- `use_time_context`
- `time_context_dim`

These are included in `train_config.json` and in `eval/infer.py` fallback config.

## Model Synchronization

Checked:

- Before edits, `train/model.py` and `eval/model.py` were identical except line-ending warnings.

Action taken:

- Implemented D01/T01 model changes in `train/model.py`.
- Mechanically copied `train/model.py` to `eval/model.py`.
- Current intended state: model files are synchronized.

## Dataset Synchronization

Checked:

- Before edits, `train/dataset.py` had extra validation split features (`time_tail`, `time_window`, `row_group_indices`, timestamp filtering) that `eval/dataset.py` did not need.
- Core feature conversion logic was otherwise aligned.

Action taken:

- Added `time_context` construction to `train/dataset.py`.
- Mechanically copied `train/dataset.py` to `eval/dataset.py` so official inference sees identical feature construction.
- This also brings train-only split helpers into eval, but `eval/infer.py` only uses `PCVRParquetDataset`, so inference behavior remains direct parquet reading.

## Checkpoint Sidecars

Confirmed in `train/trainer.py`:

- Every best checkpoint directory is intended to contain:
  - `model.pt`
  - `schema.json`
  - `train_config.json`
- `ns_groups.json` is copied when `--ns_groups_json` points to an existing file.
- If copied, `train_config.json["ns_groups_json"]` is rewritten to the basename for portability.

Risk checked:

- `trainer.py` writes sidecars only after `EarlyStopping` has actually produced `model.pt` on the AUC-selected branch, avoiding sidecar-only best directories.

## G01 Status

Current 0.809921 baseline did **not** use group tokenizer:

- It used `ns_tokenizer_type=rankmixer`.
- It used `ns_groups_json=""`.

Prepared G01 as a separate experiment:

- `--ns_tokenizer_type group`
- `--ns_groups_json "${SCRIPT_DIR}/ns_groups.json"`
- `--num_queries 1`
- `--d_model 64`
- `--user_dense_projector_type flat`
- `--use_time_context 0`
- With `num_sequences=4`, `num_ns=7 user_int groups + 1 user_dense + 4 item_int groups = 12`, so `T=1*4+12=16` and `64%16=0`.

Do not combine G01 with D01/T01 in the same official eval submission.
