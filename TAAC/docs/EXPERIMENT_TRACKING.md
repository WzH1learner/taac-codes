# EXPERIMENT TRACKING

Official eval is the decision metric. Validation AUC is only a training-health
and candidate-filtering signal.

## Baseline

| ID | Change | Config | Best Valid AUC | Valid LogLoss | Official Eval AUC | Sidecars | OOM | Conclusion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B00 | Current official best | `rankmixer, user_ns_tokens=5, item_ns_tokens=2, num_queries=2, d_model=64, seq=128/128/256/256, emb_skip_threshold=5M, BCE, flat user_dense, no time_context` | ~`0.861063@epoch6` | TBD | `0.809921` | `model.pt`, `schema.json`, `train_config.json`; no `ns_groups.json` because `ns_groups_json=""` | No | Rollback baseline |

## Single-Variable Experiments

| ID | Change | Config Delta | Best Valid AUC | Valid LogLoss | Official Eval AUC | Sidecars | OOM | Conclusion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D01 | Grouped user dense projector | `--user_dense_projector_type grouped`, keep `--use_time_context 0`, keep RankMixer 5M baseline | TBD | TBD | TBD | Check `train_config.json` has `user_dense_projector_type=grouped` | TBD | Pending |
| T01 | Time context into user_dense token | `--use_time_context 1`, keep `--user_dense_projector_type flat`, keep RankMixer 5M baseline | TBD | TBD | TBD | Check `train_config.json` has `use_time_context=true`, `time_context_dim=14` | TBD | Pending |
| T02 | Time context + D01 only after singles | Only consider after D01 and T01 official evals | TBD | TBD | TBD | TBD | TBD | Blocked until singles |
| G01 | Group tokenizer fair d64/q1 | `--ns_tokenizer_type group --ns_groups_json "${SCRIPT_DIR}/ns_groups.json" --num_queries 1 --d_model 64 --user_dense_projector_type flat --use_time_context 0`, `T=16`, `64%16=0` | TBD | TBD | TBD | Must include `ns_groups.json` in checkpoint | TBD | Pending |

## Rejected / Caution

| ID | Change | Evidence | Official Eval AUC | Conclusion |
| --- | --- | --- | --- | --- |
| Old group | `group + num_queries=1 + threshold=1M` | Valid high but official low; also changed query capacity | `0.805905` | Not a clean G01 conclusion |
| Seq x2 | Longer seq | Valid improved, official flat | `0.809907` | Not priority |
| 7M threshold | More high-cardinality embeddings | Valid did not track official | `0.806861` | Keep 5M |
| time_window=2h | Narrow sample-level valid | Removes near-test samples from training | `0.7883` | Diagnostic only |
| focal alpha 0.25/0.5 | Calibration poor, prob_mean too high | Not worth eval unless explicit ablation | TBD | Avoid for score push |

## Submission Checklist

- Confirm `MODEL_OUTPUT_PATH` points at a directory containing `model.pt`.
- Confirm checkpoint contains `train_config.json` and `schema.json`.
- For G01, confirm checkpoint also contains `ns_groups.json`.
- Confirm `eval/infer.py` and `eval/model.py` are synchronized with training-side structural changes.
- Record both validation metrics and official eval AUC before drawing conclusions.
