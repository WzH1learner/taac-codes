#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"
# 【新增】启用 PyTorch 显存碎片优化，防止因碎片导致分配失败
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# ---- Active config: rollback official-best backbone (flat dense, no time context) ----
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --user_dense_projector_type flat \
    --use_time_context 0 \
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
    --amp \
    --amp_dtype bfloat16 \
    "$@"

# ---- D01 single-variable experiment: grouped user_dense projector ----
# Add to the active block above:
#     --user_dense_projector_type grouped
#
# ---- T01/T02 single-variable experiment: time context into user_dense token ----
# Add to the active block above:
#     --use_time_context 1
#
# ---- G01 single-variable experiment: GroupNSTokenizer driven by ns_groups.json ----
# Uses feature grouping from ns_groups.json (7 user groups + 4 item groups).
# With d_model=64 and num_ns=12 (7 user_int + 1 user_dense + 4 item_int),
# only num_queries=1 satisfies d_model % T == 0 (T = num_queries*4 + num_ns = 16).
# Keep --user_dense_projector_type flat and --use_time_context 0 here.
# To switch, comment out the block above and uncomment the block below.
#
# python3 -u "${SCRIPT_DIR}/train.py" \
#     --ns_tokenizer_type group \
#     --ns_groups_json "${SCRIPT_DIR}/ns_groups.json" \
#     --num_queries 1 \
#     --d_model 64 \
#     --user_dense_projector_type flat \
#     --use_time_context 0 \
#     --emb_skip_threshold 5000000 \
#     --num_workers 8 \
#     "$@"
