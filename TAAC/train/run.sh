#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- Active config: D01 current official-best backbone ----
# D01_grouped_user_dense_single_token:
#   official eval AUC = 0.818095
#   Only structural change vs E00 transformer baseline: user_dense_projector_type=grouped.
python3 -u "${SCRIPT_DIR}/train.py" \
    --seq_encoder_type transformer \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --user_dense_projector_type grouped \
    --use_time_context 0 \
    --use_seq_recent_stats 0 \
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

# ---- D02 single-variable experiment: grouped dense + SwiGLU ----
# Purpose: test whether D01 grouped user_dense also stacks with the historical
# SwiGLU sequence encoder. The only variable vs active D01 is:
#     --seq_encoder_type transformer -> swiglu
#
# python3 -u "${SCRIPT_DIR}/train.py" \
#     --seq_encoder_type swiglu \
#     --ns_tokenizer_type rankmixer \
#     --user_ns_tokens 5 \
#     --item_ns_tokens 2 \
#     --num_queries 2 \
#     --ns_groups_json "" \
#     --user_dense_projector_type grouped \
#     --use_time_context 0 \
#     --use_seq_recent_stats 0 \
#     --seq_recent_stats_gate_init 0.1 \
#     --emb_skip_threshold 5000000 \
#     --batch_size 128 \
#     --num_workers 8 \
#     --seq_max_lens "seq_a:128,seq_b:128,seq_c:256,seq_d:256" \
#     --num_epochs 20 \
#     --sparse_lr 0.05 \
#     --dropout_rate 0.01 \
#     --patience 15 \
#     --reinit_sparse_after_epoch 0 \
#     --reinit_cardinality_threshold 0 \
#     --loss_type bce \
#     --amp \
#     --amp_dtype bfloat16 \
#     "$@"
#
# Decision rule:
#   D02 > 0.818095: switch mainline to swiglu + grouped dense.
#   0.816 <= D02 <= 0.818095: keep D01 as mainline; D02 is near-tie.
#   D02 < 0.816: keep transformer + grouped dense.

# ---- R01 single-variable experiment: D01 + seq recent stats residual ----
# Purpose: test relative recency / recent-window intensity without reusing T01
# root time_context and without adding RankMixer tokens. The only variable vs
# active D01 is:
#     --use_seq_recent_stats 0 -> 1
#
# python3 -u "${SCRIPT_DIR}/train.py" \
#     --seq_encoder_type transformer \
#     --ns_tokenizer_type rankmixer \
#     --user_ns_tokens 5 \
#     --item_ns_tokens 2 \
#     --num_queries 2 \
#     --ns_groups_json "" \
#     --user_dense_projector_type grouped \
#     --use_time_context 0 \
#     --use_seq_recent_stats 1 \
#     --seq_recent_stats_gate_init 0.1 \
#     --emb_skip_threshold 5000000 \
#     --batch_size 128 \
#     --num_workers 8 \
#     --seq_max_lens "seq_a:128,seq_b:128,seq_c:256,seq_d:256" \
#     --num_epochs 20 \
#     --sparse_lr 0.05 \
#     --dropout_rate 0.01 \
#     --patience 15 \
#     --reinit_sparse_after_epoch 0 \
#     --reinit_cardinality_threshold 0 \
#     --loss_type bce \
#     --amp \
#     --amp_dtype bfloat16 \
#     "$@"
#
# Eval rule:
#   valid AUC >= 0.864399: prioritize official eval unless D02 is stronger.
#   valid AUC near D01 with healthier LogLoss/Brier/prob_mean: keep as candidate.
#   clearly lower valid or bad calibration: do not spend official eval.

# ---- G01 archived candidate: GroupNSTokenizer driven by ns_groups.json ----
# Not next priority. If retried, keep it independent from D01/T01 and make the
# seq encoder explicit. With d_model=64 and num_ns=12, num_queries=1 gives T=16.
#
# python3 -u "${SCRIPT_DIR}/train.py" \
#     --seq_encoder_type transformer \
#     --ns_tokenizer_type group \
#     --ns_groups_json "${SCRIPT_DIR}/ns_groups.json" \
#     --num_queries 1 \
#     --d_model 64 \
#     --user_dense_projector_type flat \
#     --use_time_context 0 \
#     --use_seq_recent_stats 0 \
#     --seq_recent_stats_gate_init 0.1 \
#     --emb_skip_threshold 5000000 \
#     --batch_size 128 \
#     --num_workers 8 \
#     --seq_max_lens "seq_a:128,seq_b:128,seq_c:256,seq_d:256" \
#     --num_epochs 20 \
#     --sparse_lr 0.05 \
#     --dropout_rate 0.01 \
#     --patience 15 \
#     --reinit_sparse_after_epoch 0 \
#     --reinit_cardinality_threshold 0 \
#     --loss_type bce \
#     --amp \
#     --amp_dtype bfloat16 \
#     "$@"
