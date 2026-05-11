# TAAC PCVRHyFormer Handoff

This document is a working handoff for continuing the TAAC PCVRHyFormer
experiments. It records the current best baseline, what has already been tried,
where the important files are, and what should be done next.

## 1. Current Goal

The goal is to improve official eval AUC, not to optimize validation AUC in
isolation.

Current best official eval:

```text
official eval AUC = 0.809921
```

Current rollback baseline:

```text
seq_encoder_type = swiglu
ns_tokenizer_type = rankmixer
emb_skip_threshold = 5000000
seq_max_lens = seq_a:128,seq_b:128,seq_c:256,seq_d:256
loss_type = bce
sparse_lr = 0.05
dropout_rate = 0.01
reinit_sparse_after_epoch = 0
reinit_cardinality_threshold = 0
valid_split_mode = tail
checkpoint_select_metric = auc
```

## 2. Main Experiment Conclusions

### Useful Or Still Plausible

1. `emb_skip_threshold=5000000` is better than the earlier `1000000` baseline.
2. `seq_encoder_type=swiglu` is better than `transformer` in our official eval.
3. Full sparse reinit is important:

```text
reinit_sparse_after_epoch = 0
reinit_cardinality_threshold = 0
```

4. `seq*2` improves validation AUC but almost does not improve official eval,
   while inference becomes much slower. It is not the next priority.
5. The most useful pending official eval candidate is the optimizer run:

```text
sparse_lr = 0.08
```

Its validation AUC is not the highest, but calibration is healthy:

```text
best valid AUC = 0.861314@epoch6
valid LogLoss = 0.224980
valid prob_mean = 0.096718
valid label_mean = 0.096785
```

### Low-Priority Or Negative Directions

Do not spend more official eval quota on these unless there is a new reason.

```text
time_window=2h:
  official eval AUC = 0.7883
  conclusion = bad; it removes the nearest timestamp samples from training.

train_include_valid=True + checkpoint_select_metric=last:
  official eval AUC = 0.80387
  conclusion = bad; validation AUC is inflated because validation rows are also
  in training.

use_seq_time_delta_proj=True:
  official eval AUC = 0.809214
  conclusion = this specific scalar per-token time-delta projection did not
  beat the 0.809921 baseline.

Transformer + 5M:
  official eval AUC = 0.80656
  conclusion = worse than SwiGLU.

emb_skip_threshold=7M:
  official eval AUC = 0.806861
  conclusion = worse than 5M.
```

Focal loss note:

```text
focal_alpha=0.25, focal_gamma=2:
  best valid AUC = 0.861691@epoch8
  valid prob_mean ~= 0.17 - 0.18
  valid label_mean ~= 0.0968
  valid LogLoss ~= 0.273 - 0.285
```

The AUC is acceptable, but calibration is badly high. Do not official-eval this
checkpoint for now. If focal is revisited, try a lower alpha and check LogLoss /
prob_mean before spending eval quota.

## 3. Project Directory

```text
train/
  train.py
    Training entry point. CLI experiment parameters are defined here.

  dataset.py
    Training data pipeline, RowGroup split, time_window split,
    train_include_valid, sequence timestamp features.

  model.py
    PCVRHyFormer model definition.

  trainer.py
    Training loop, loss, checkpoint saving, sparse optimizer, sparse reinit.

  run.sh
    Platform training script. Most platform experiments are launched by changing
    this file's arguments.

  ns_groups.json
    NS grouping config.

eval/
  infer.py
    Official platform inference entry point. It rebuilds the model from
    train_config.json and must stay compatible with training.

  dataset.py
    Eval data pipeline.

  model.py
    Eval-side model definition. Structural model changes usually need to be
    mirrored from train/model.py.

research/code/
  01_external_repos_baseline_notes.md
    Experiment ledger, external high-score experience, current priorities.

  inspect_pcvr_structure.py
    Local structure/data inspection helper.

README_baseline.md
  Current baseline notes, experiment state, and TODO.

parse_log.py
  Local log parser for platform training logs.

.gitignore
  Ignores checkpoints, logs, notebooks, and colleague-only files.
```

## 4. Train/Eval Compatibility Rules

If a model structure or model input changes, update both training and eval.

Usually paired files:

```text
train/model.py  <-> eval/model.py
train/dataset.py <-> eval/dataset.py
train/train.py structural args
eval/infer.py fallback config / train_config parsing
```

Examples that require eval compatibility:

```text
new model constructor argument
new ModelInput field
new sequence feature tensor
new structural hyperparameter saved in train_config.json
```

Training-only controls usually do not require eval changes:

```text
valid_split_mode
valid_time_window_hours
train_include_valid
checkpoint_select_metric
patience
num_epochs
```

## 5. Files Not To Share Or Commit

These are local logs or colleague-owned analysis exports:

```text
all_NOTEBOOK.py
*.ipynb
7embeddingskip.txt
seq_log.txt
validation.txt
yuyihun.txt
```

They are ignored by `.gitignore`.

## 6. Recommended Next Steps

### Step 1: Official Eval The Sparse-LR Experiment

Submit / retrieve official eval for:

```text
seq_encoder_type = swiglu
ns_tokenizer_type = rankmixer
emb_skip_threshold = 5000000
seq_max_lens = seq_a:128,seq_b:128,seq_c:256,seq_d:256
loss_type = bce
sparse_lr = 0.08
dropout_rate = 0.01
reinit_sparse_after_epoch = 0
reinit_cardinality_threshold = 0
valid_split_mode = tail
checkpoint_select_metric = auc
```

If official eval beats `0.809921`, continue optimizer direction:

```text
sparse_lr = 0.10
lr warmup
scale
dense lr tuning
```

If it does not beat `0.809921`, do not keep doing pure sparse-lr search.

### Step 2: Revisit External High-Score Experience

External high-score clues that still look important:

```text
scale + lr warmup
optimizer details
item-side feature handling
pair features
sequence behavior timestamp features
focal loss
```

Current priority:

```text
optimizer / warmup / scale
> item-side features
> pair features
> sequence recency / interval aggregate features
> lower-alpha focal
```

For time features, avoid simply adding hour / week buckets or one scalar token
delta. Better hypotheses:

```text
recent-window counts per sequence domain
last-action recency
min / mean / max interval
recent behavior intensity
item versus historical sequence side-info match statistics
domain-level recency aggregation
```

## 7. Latest Code Checkpoint

The cleaned-up git commit before this handoff was:

```text
2d06167 Add experiment controls and TAAC notes
```

That commit includes:

```text
valid split controls
train_include_valid
checkpoint_select_metric
optional sequence time-delta projection
eval-side compatibility
parse_log.py
updated README / research notes
```

The current best model to fall back to is still:

```text
official eval AUC = 0.809921
RankMixer + SwiGLU + 5M + short seq + BCE + sparse_lr=0.05
```
