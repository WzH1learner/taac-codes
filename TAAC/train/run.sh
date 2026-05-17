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
    --use_pair_dense 0 \
    --pair_dense_gate_init 0.05 \
    --pair_dense_pairs_json "" \
    --use_target_matched_recency 0 \
    --target_matched_recency_gate_init 0.005 \
    --target_matched_recency_feature_mode any_only \
    --target_matched_recency_pairs_windows_json "" \
    --use_seq_target_match_flags 0 \
    --seq_target_match_flag_gate_init 0.01 \
    --seq_target_match_flag_domain seq_d \
    --seq_target_match_flag_specs_json "" \
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

# ---- F00 fast-screening timing / compile status ----
# Default active D01 stays no-compile. Extra args are appended through "$@".
#
# D01 no-compile timing:
#   bash TAAC/train/run.sh
#
# F00b_D01_compile_baseline with:
#   --torch_compile --compile_mode reduce-overhead
# failed on cloud with a PyTorch Inductor CUDA graph allocator error:
#   RuntimeError: Expected curr_block->next == nullptr ...
# Treat reduce-overhead compile as rejected/invalid for this environment.
# Do not use compile metrics as a screening baseline.

# ---- A01 candidate: aligned user_int/user_dense weighted pooling ----
# Purpose: use same-fid user_dense values as weights over aligned user_int
# embeddings. Only variable vs active D01:
#     --use_aligned_user_int_dense 0 -> 1
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
#     --use_seq_recent_stats 0 \
#     --seq_recent_stats_gate_init 0.1 \
#     --use_pair_dense 0 \
#     --pair_dense_gate_init 0.05 \
#     --pair_dense_pairs_json "" \
#     --use_target_matched_recency 0 \
#     --target_matched_recency_gate_init 0.005 \
#     --target_matched_recency_feature_mode any_only \
#     --target_matched_recency_pairs_windows_json "" \
#     --use_seq_target_match_flags 0 \
#     --seq_target_match_flag_gate_init 0.01 \
#     --seq_target_match_flag_domain seq_d \
#     --seq_target_match_flag_specs_json "" \
#     --use_aligned_user_int_dense 1 \
#     --aligned_user_int_dense_gate_init 0.05 \
#     --aligned_user_int_dense_fids_json "" \
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

# ---- D02 rejected: grouped dense + SwiGLU ----
# Clean result: valid AUC=0.864255, LogLoss=0.223297, official AUC=0.81476.
# Conclusion: swiglu + grouped dense is weaker than D01; keep archived only.
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
#     --use_pair_dense 0 \
#     --pair_dense_gate_init 0.05 \
#     --pair_dense_pairs_json "" \
#     --use_target_matched_recency 0 \
#     --target_matched_recency_gate_init 0.005 \
#     --target_matched_recency_feature_mode any_only \
#     --target_matched_recency_pairs_windows_json "" \
#     --use_seq_target_match_flags 0 \
#     --seq_target_match_flag_gate_init 0.01 \
#     --seq_target_match_flag_domain seq_d \
#     --seq_target_match_flag_specs_json "" \
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
# ---- R01 rejected: D01 + global seq recent stats residual ----
# Clean result: official AUC=0.808144.
# Conclusion: do not continue R01/time_context/global recency.
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
#     --use_pair_dense 0 \
#     --pair_dense_gate_init 0.05 \
#     --pair_dense_pairs_json "" \
#     --use_target_matched_recency 0 \
#     --target_matched_recency_gate_init 0.005 \
#     --target_matched_recency_feature_mode any_only \
#     --target_matched_recency_pairs_windows_json "" \
#     --use_seq_target_match_flags 0 \
#     --seq_target_match_flag_gate_init 0.01 \
#     --seq_target_match_flag_domain seq_d \
#     --seq_target_match_flag_specs_json "" \
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
# ---- P2 candidate: D01 + pair/time exact-match dense residual ----
# Purpose: use stable target-history match signal found by pair_match_eda v2.
# Only variable vs active D01:
#     --use_pair_dense 0 -> 1
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
#     --use_seq_recent_stats 0 \
#     --seq_recent_stats_gate_init 0.1 \
#     --use_pair_dense 1 \
#     --pair_dense_gate_init 0.05 \
#     --pair_dense_pairs_json "" \
#     --use_target_matched_recency 0 \
#     --target_matched_recency_gate_init 0.005 \
#     --target_matched_recency_feature_mode any_only \
#     --target_matched_recency_pairs_windows_json "" \
#     --use_seq_target_match_flags 0 \
#     --seq_target_match_flag_gate_init 0.01 \
#     --seq_target_match_flag_domain seq_d \
#     --seq_target_match_flag_specs_json "" \
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

# ---- P3a candidate: D01 + target-matched recency any-only residual ----
# Purpose: use EDA v3 stable target-matched recency signals without adding
# global recency, log_count, last_gap, bare match_any, or RankMixer tokens.
# Only variable vs active D01:
#     --use_target_matched_recency 0 -> 1
#
# Platform command:
#   bash TAAC/train/run.sh \
#     --use_target_matched_recency 1 \
#     --target_matched_recency_gate_init 0.005 \
#     --target_matched_recency_feature_mode any_only \
#     --target_matched_recency_pairs_windows_json ""
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
#     --use_seq_recent_stats 0 \
#     --seq_recent_stats_gate_init 0.1 \
#     --use_pair_dense 0 \
#     --pair_dense_gate_init 0.05 \
#     --pair_dense_pairs_json "" \
#     --use_aligned_user_int_dense 0 \
#     --aligned_user_int_dense_gate_init 0.05 \
#     --aligned_user_int_dense_fids_json "" \
#     --use_target_matched_recency 1 \
#     --target_matched_recency_gate_init 0.005 \
#     --target_matched_recency_feature_mode any_only \
#     --target_matched_recency_pairs_windows_json "" \
#     --use_seq_target_match_flags 0 \
#     --seq_target_match_flag_gate_init 0.01 \
#     --seq_target_match_flag_domain seq_d \
#     --seq_target_match_flag_specs_json "" \
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

# ---- P4 candidate: seq_d target-match token flags ----
# Purpose: inject target-history match signals into seq_d token embeddings so
# the transformer can decide how to use matched tokens. This does not add
# RankMixer tokens and does not touch final_repr residuals.
# Only variable vs active D01:
#     --use_seq_target_match_flags 0 -> 1
#
# Platform command:
#   bash TAAC/train/run.sh \
#     --use_seq_target_match_flags 1 \
#     --seq_target_match_flag_gate_init 0.01 \
#     --seq_target_match_flag_specs_json ""
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
#     --use_seq_recent_stats 0 \
#     --seq_recent_stats_gate_init 0.1 \
#     --use_pair_dense 0 \
#     --pair_dense_gate_init 0.05 \
#     --pair_dense_pairs_json "" \
#     --use_aligned_user_int_dense 0 \
#     --aligned_user_int_dense_gate_init 0.05 \
#     --aligned_user_int_dense_fids_json "" \
#     --use_target_matched_recency 0 \
#     --target_matched_recency_gate_init 0.005 \
#     --target_matched_recency_feature_mode any_only \
#     --target_matched_recency_pairs_windows_json "" \
#     --use_seq_target_match_flags 1 \
#     --seq_target_match_flag_gate_init 0.01 \
#     --seq_target_match_flag_domain seq_d \
#     --seq_target_match_flag_specs_json "" \
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
#     --use_pair_dense 0 \
#     --pair_dense_gate_init 0.05 \
#     --pair_dense_pairs_json "" \
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
