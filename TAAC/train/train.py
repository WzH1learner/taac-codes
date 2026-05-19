"""PCVRHyFormer training entry point (self-contained baseline).

Usage:
    python train.py [--num_epochs 10] [--batch_size 256] ...

Environment variables (take precedence over CLI flags):
    TRAIN_DATA_PATH  Training data directory (*.parquet + schema.json)
    TRAIN_CKPT_PATH  Checkpoint output directory
    TRAIN_LOG_PATH   Log directory
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Any, List, Tuple

import torch

from utils import set_seed, EarlyStopping, create_logger
from dataset import (
    DEFAULT_PAIR_DENSE_PAIRS,
    DEFAULT_SEQ_TARGET_MATCH_FLAG_SPECS,
    DEFAULT_TARGET_MATCHED_RECENCY_PAIRS_WINDOWS,
    FeatureSchema,
    NUM_TIME_BUCKETS,
    PAIR_DENSE_FEATS_PER_PAIR,
    SEQ_RECENT_STATS_DIM,
    get_pcvr_data,
)
from model import PCVRHyFormer
from trainer import PCVRHyFormerRankingTrainer


DEFAULT_ALIGNED_USER_INT_DENSE_FIDS = [62, 63, 64, 65, 66, 89, 90, 91]
DEFAULT_SEQ_D_IMPORTANT_SIDE_FIDS = [25, 24]


def build_feature_specs(
    schema: FeatureSchema,
    per_position_vocab_sizes: List[int],
) -> List[Tuple[int, int, int]]:
    """Build feature_specs of the form ``[(vocab_size, offset, length), ...]``
    ordered by the positions recorded in ``schema.entries``.
    """
    specs: List[Tuple[int, int, int]] = []
    for fid, offset, length in schema.entries:
        vs = max(per_position_vocab_sizes[offset:offset + length])
        specs.append((vs, offset, length))
    return specs


def load_pair_dense_pairs(pair_dense_pairs_json: str) -> List[List[Any]]:
    if not pair_dense_pairs_json:
        return DEFAULT_PAIR_DENSE_PAIRS
    if os.path.exists(pair_dense_pairs_json):
        with open(pair_dense_pairs_json, 'r', encoding='utf-8') as f:
            return json.load(f)
    return json.loads(pair_dense_pairs_json)


def load_aligned_user_int_dense_fids(fids_json: str) -> List[int]:
    if not fids_json:
        return DEFAULT_ALIGNED_USER_INT_DENSE_FIDS
    if os.path.exists(fids_json):
        with open(fids_json, 'r', encoding='utf-8') as f:
            values = json.load(f)
    else:
        values = json.loads(fids_json)
    return [int(v) for v in values]


def load_target_matched_recency_pairs_windows(specs_json: str) -> List[dict]:
    if not specs_json:
        return DEFAULT_TARGET_MATCHED_RECENCY_PAIRS_WINDOWS
    if os.path.exists(specs_json):
        with open(specs_json, 'r', encoding='utf-8') as f:
            values = json.load(f)
    else:
        values = json.loads(specs_json)
    normalized = []
    for spec in values:
        normalized.append({
            "item_fid": int(spec["item_fid"]),
            "domain": str(spec["domain"]),
            "side_fid": int(spec["side_fid"]),
            "windows": [str(w) for w in spec.get("windows", [])],
        })
    return normalized


def load_seq_target_match_flag_specs(specs_json: str) -> List[dict]:
    if not specs_json:
        return DEFAULT_SEQ_TARGET_MATCH_FLAG_SPECS
    if os.path.exists(specs_json):
        with open(specs_json, 'r', encoding='utf-8') as f:
            values = json.load(f)
    else:
        values = json.loads(specs_json)
    normalized = []
    for spec in values:
        normalized.append({
            "item_fid": int(spec["item_fid"]),
            "domain": str(spec["domain"]),
            "side_fid": int(spec["side_fid"]),
        })
    return normalized


def load_seq_d_important_side_fids(fids_json: str) -> List[int]:
    if not fids_json:
        return DEFAULT_SEQ_D_IMPORTANT_SIDE_FIDS
    if os.path.exists(fids_json):
        with open(fids_json, 'r', encoding='utf-8') as f:
            values = json.load(f)
    else:
        values = json.loads(fids_json)
    return [int(v) for v in values]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PCVRHyFormer Training")

    # Paths (environment variables take precedence).
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Training data directory (env: TRAIN_DATA_PATH)')
    parser.add_argument('--schema_path', type=str, default=None,
                        help='Schema JSON path (defaults to <data_dir>/schema.json)')
    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help='Checkpoint output directory (env: TRAIN_CKPT_PATH)')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Log directory (env: TRAIN_LOG_PATH)')

    # Training hyperparameters.
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for both training and validation')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for dense parameters (AdamW)')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Maximum number of training epochs '
                             '(typically terminated earlier by early stopping)')
    parser.add_argument('--patience', type=int, default=3,
                        help='Early-stopping patience '
                             '(number of validations without improvement)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Training device, e.g. cuda or cpu')
    parser.add_argument('--amp', action='store_true', default=False,
                        help='Enable CUDA automatic mixed precision for training/eval')
    parser.add_argument('--amp_dtype', type=str, default='bfloat16',
                        choices=['bfloat16', 'float16'],
                        help='AMP dtype when --amp is enabled')
    parser.add_argument('--torch_compile', action='store_true', default=False,
                        help='Compile model.forward with torch.compile when available')
    parser.add_argument('--compile_mode', type=str, default='default',
                        choices=['default', 'reduce-overhead', 'max-autotune'],
                        help='torch.compile mode when --torch_compile is enabled')

    # Data pipeline.
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of DataLoader workers')
    parser.add_argument('--buffer_batches', type=int, default=20,
                        help='Shuffle buffer size, in units of batches. '
                             'Lower values reduce memory usage.')
    parser.add_argument('--train_ratio', type=float, default=1.0,
                        help='Fraction of training Row Groups to use (takes the first N%)')
    parser.add_argument('--valid_ratio', type=float, default=0.1,
                        help='Fraction of all Row Groups used for validation (takes the tail)')
    parser.add_argument('--valid_split_mode', type=str, default='tail',
                        choices=['tail', 'head', 'middle', 'random', 'time_tail', 'time_window'],
                        help='Which Row Groups to reserve for validation. '
                             'tail keeps the historical behavior; time_tail sorts '
                             'Row Groups by timestamp; random/head/middle are '
                             'diagnostics for train/valid/test distribution drift; '
                             'time_window uses sample-level timestamp filtering.')
    parser.add_argument('--valid_time_window_hours', type=float, default=0.0,
                        help='When --valid_split_mode=time_window, use the last '
                             'N hours by sample timestamp as validation and all '
                             'earlier samples as training. Example: 2.0 roughly '
                             'matches the official test span observed on 2026-05-02.')
    parser.add_argument('--train_include_valid', action='store_true', default=False,
                        help='Let the training loader include validation rows. '
                             'The validation loader is still built and logged, '
                             'but its metrics become monitoring signals rather '
                             'than strict holdout metrics.')
    parser.add_argument('--checkpoint_select_metric', type=str, default='auc',
                        choices=['auc', 'logloss', 'auc_then_logloss'],
                        help='Checkpoint ranking metric. auc keeps the historical '
                             'best-valid-AUC behavior; logloss prefers lower valid '
                             'LogLoss; auc_then_logloss treats AUC values within '
                             '0.0003 as ties, then uses LogLoss/Brier/prob-gap.')
    parser.add_argument('--keep_top_k_checkpoints', type=int, default=1,
                        help='Keep top-K validation checkpoints. Default 1 preserves '
                             'the historical single best_model behavior.')
    parser.add_argument('--always_save_last_checkpoint', type=int, default=0,
                        choices=[0, 1],
                        help='Also keep a rolling last checkpoint after each epoch/validation.')
    parser.add_argument('--eval_every_n_steps', type=int, default=0,
                        help='Run validation every N steps '
                             '(0 = only at the end of each epoch)')
    parser.add_argument('--seq_max_lens', type=str,
                        default='seq_a:256,seq_b:256,seq_c:512,seq_d:512',
                        help='Per-domain sequence truncation, format: seq_d:256,seq_c:128')

    # Model hyperparameters.
    parser.add_argument('--d_model', type=int, default=64,
                        help='Backbone hidden dimension (output size of each block)')
    parser.add_argument('--emb_dim', type=int, default=64,
                        help='Per-Embedding-table dimension (before projection)')
    parser.add_argument('--num_queries', type=int, default=2,
                        help='Number of Query tokens generated independently per sequence domain')
    parser.add_argument('--num_hyformer_blocks', type=int, default=2,
                        help='Number of stacked MultiSeqHyFormerBlock layers')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads (must satisfy d_model %% num_heads == 0)')
    parser.add_argument('--seq_encoder_type', type=str, default='transformer',
                        choices=['swiglu', 'transformer', 'longer'],
                        help='Sequence encoder variant: '
                             'swiglu = SwiGLU without attention, '
                             'transformer = standard self-attention, '
                             'longer = Top-K compressed encoder '
                             '(only this variant consumes --seq_top_k / --seq_causal)')
    parser.add_argument('--hidden_mult', type=int, default=4,
                        help='FFN inner-dim multiplier relative to d_model')
    parser.add_argument('--dropout_rate', type=float, default=0.01,
                        help='Dropout rate for the backbone '
                             '(seq id-embedding dropout is twice this value)')
    parser.add_argument('--seq_top_k', type=int, default=50,
                        help='Number of most-recent tokens kept by LongerEncoder '
                             '(only effective when --seq_encoder_type=longer)')
    parser.add_argument('--seq_causal', action='store_true', default=False,
                        help='Whether the LongerEncoder self-attention uses a causal mask '
                             '(only effective when --seq_encoder_type=longer)')
    parser.add_argument('--action_num', type=int, default=1,
                        help='Classifier output dimension '
                             '(1 = single binary-classification logit; >1 = multi-label)')
    parser.add_argument('--use_time_buckets', action='store_true', default=True,
                        help='Enable the time-bucket embedding (default on). '
                             'The actual bucket count is uniquely determined by '
                             'dataset.BUCKET_BOUNDARIES; this flag is a pure on/off switch.')
    parser.add_argument('--no_time_buckets', dest='use_time_buckets', action='store_false',
                        help='Disable the time-bucket embedding')
    parser.add_argument('--rank_mixer_mode', type=str, default='full',
                        choices=['full', 'ffn_only', 'none'],
                        help='RankMixerBlock mode: '
                             'full = token mixing + per-token FFN (requires d_model divisible by T), '
                             'ffn_only = per-token FFN only, '
                             'none = identity passthrough')
    parser.add_argument('--use_rope', action='store_true', default=False,
                        help='Enable RoPE positional encoding in sequence attention')
    parser.add_argument('--rope_base', type=float, default=10000.0,
                        help='RoPE base frequency (default 10000)')
    parser.add_argument('--use_seq_time_delta_proj', action='store_true', default=False,
                        help='Add a dense projection of normalized log timestamp '
                             'gap to every sequence token, in addition to the '
                             'existing discrete time-bucket embedding.')
    parser.add_argument('--use_time_context', type=int, default=0, choices=[0, 1],
                        help='Add root/domain time summary context into the single user_dense token')
    parser.add_argument('--use_seq_recent_stats', type=int, default=0, choices=[0, 1],
                        help='Add relative per-sequence recency/intensity stats as a residual '
                             'to the final representation')
    parser.add_argument('--seq_recent_stats_gate_init', type=float, default=0.1,
                        help='Initial scalar residual gate for --use_seq_recent_stats')
    parser.add_argument('--use_pair_dense', type=int, default=0, choices=[0, 1],
                        help='Add selected target-history exact-match features as a residual '
                             'to the final representation')
    parser.add_argument('--pair_dense_gate_init', type=float, default=0.05,
                        help='Initial scalar residual gate for --use_pair_dense')
    parser.add_argument('--pair_dense_pairs_json', type=str, default='',
                        help='Path to or inline JSON list of [item_fid, domain, side_fid] '
                             'pairs; empty uses built-in P2 v1 pairs')
    parser.add_argument('--use_aligned_user_int_dense', type=int, default=0, choices=[0, 1],
                        help='Use same-fid user_dense values to weighted-pool aligned '
                             'user_int embeddings as a final residual')
    parser.add_argument('--aligned_user_int_dense_gate_init', type=float, default=0.05,
                        help='Initial scalar residual gate for --use_aligned_user_int_dense')
    parser.add_argument('--aligned_user_int_dense_fids_json', type=str, default='',
                        help='Path to or inline JSON list of aligned fids; empty uses '
                             'the default A01 fids')
    parser.add_argument('--use_target_matched_recency', type=int, default=0, choices=[0, 1],
                        help='Add target-matched recency any features as a final residual')
    parser.add_argument('--target_matched_recency_gate_init', type=float, default=0.005,
                        help='Initial scalar residual gate for --use_target_matched_recency')
    parser.add_argument('--target_matched_recency_pairs_windows_json', type=str, default='',
                        help='Path to or inline JSON list of P3 pair/window specs; '
                             'empty uses built-in P3a lite specs')
    parser.add_argument('--target_matched_recency_feature_mode', type=str, default='any_only',
                        choices=['any_only'],
                        help='P3 feature mode. P3a supports any_only only.')
    parser.add_argument('--use_seq_target_match_flags', type=int, default=0, choices=[0, 1],
                        help='Add target-match flag embeddings to sequence tokens')
    parser.add_argument('--seq_target_match_flag_specs_json', type=str, default='',
                        help='Path to or inline JSON list of P4 specs; empty uses built-in specs')
    parser.add_argument('--seq_target_match_flag_gate_init', type=float, default=0.01,
                        help='Initial scalar token-level gate for --use_seq_target_match_flags')
    parser.add_argument('--seq_target_match_flag_domain', type=str, default='seq_d',
                        help='Sequence domain that receives target-match flag embeddings')
    parser.add_argument('--seq_d_side_projector_type', type=str, default='flat',
                        choices=['flat', 'grouped'],
                        help='Optional grouped sideinfo projector for seq_d only')
    parser.add_argument('--seq_d_important_side_fids_json', type=str, default='',
                        help='Path to or inline JSON list of important seq_d side fids; empty uses [25,24]')
    parser.add_argument('--user_dense_projector_type', type=str, default='flat',
                        choices=['flat', 'grouped'],
                        help='flat keeps the legacy concat projection; grouped uses fid-aware dense branches')

    # Loss function.
    parser.add_argument('--loss_type', type=str, default='bce', choices=['bce', 'focal'],
                        help='Loss type: bce = BCEWithLogits, focal = Focal Loss')
    parser.add_argument('--focal_alpha', type=float, default=0.1,
                        help='Focal Loss positive-class weight alpha '
                             '(effective only when --loss_type=focal)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal Loss focusing parameter gamma '
                             '(effective only when --loss_type=focal)')

    # Sparse optimizer.
    parser.add_argument('--sparse_lr', type=float, default=0.05,
                        help='Learning rate for sparse parameters (Adagrad over Embeddings)')
    parser.add_argument('--sparse_weight_decay', type=float, default=0.0,
                        help='Weight decay for sparse parameters (Adagrad over Embeddings)')
    parser.add_argument('--reinit_sparse_after_epoch', type=int, default=2,
                        help='Starting from the N-th epoch, at the end of every epoch '
                             're-initialize Embeddings with vocab_size > '
                             '--reinit_cardinality_threshold and rebuild the Adagrad '
                             'optimizer state (cold-restart trick for high-cardinality '
                             'features to reduce overfitting)')
    parser.add_argument('--reinit_cardinality_threshold', type=int, default=100000,
                        help='Cardinality threshold used by the re-init strategy: '
                             'Embeddings whose vocab_size exceeds this value are reset '
                             'at each epoch end (0 = reset every Embedding with '
                             'vocab_size > 0; use a very large threshold or set '
                             '--reinit_sparse_after_epoch beyond --num_epochs to disable)')

    # Embedding construction control.
    parser.add_argument('--emb_skip_threshold', type=int, default=5000000,
                        help='At model construction time, features whose vocab_size '
                             'exceeds this value get no Embedding and are represented '
                             'by a zero vector at forward time (0 = no skipping; '
                             'all features get an Embedding). Useful for saving GPU '
                             'memory on ultra-high-cardinality features.')
    parser.add_argument('--seq_id_threshold', type=int, default=10000,
                        help='Within the sequence tokenizer, features with vocab_size '
                             'exceeding this value are treated as id features and receive '
                             'extra dropout(rate*2) during training to reduce overfitting. '
                             'Features at or below this threshold are treated as side-info '
                             'and receive no extra dropout.')

    _default_ns_groups = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'ns_groups.json')
    parser.add_argument('--ns_groups_json', type=str, default=_default_ns_groups,
                        help='Path to the NS-groups JSON file. If it does not exist, '
                             'each feature is placed in its own singleton group.')

    # NS tokenizer variant.
    parser.add_argument('--ns_tokenizer_type', type=str, default='rankmixer',
                        choices=['group', 'rankmixer'],
                        help='NS tokenizer variant: '
                             'group = project each group to one token, '
                             'rankmixer = concatenate all embeddings then split into '
                             'equal-size chunks (token count is tunable)')
    parser.add_argument('--user_ns_tokens', type=int, default=5,
                        help='Number of user NS tokens in rankmixer mode '
                             '(0 = automatically use the number of user groups)')
    parser.add_argument('--item_ns_tokens', type=int, default=2,
                        help='Number of item NS tokens in rankmixer mode '
                             '(0 = automatically use the number of item groups)')

    args = parser.parse_args()
    args.seq_recent_stats_dim = SEQ_RECENT_STATS_DIM
    args.pair_dense_pairs = load_pair_dense_pairs(args.pair_dense_pairs_json)
    args.pair_dense_dim = len(args.pair_dense_pairs) * PAIR_DENSE_FEATS_PER_PAIR
    args.aligned_user_int_dense_fids = load_aligned_user_int_dense_fids(
        args.aligned_user_int_dense_fids_json)
    args.target_matched_recency_pairs_windows = (
        load_target_matched_recency_pairs_windows(
            args.target_matched_recency_pairs_windows_json))
    args.target_matched_recency_dim = sum(
        len(spec["windows"]) for spec in args.target_matched_recency_pairs_windows)
    args.seq_target_match_flag_specs = load_seq_target_match_flag_specs(
        args.seq_target_match_flag_specs_json)
    args.seq_target_match_flag_num_flags = sum(
        1 for spec in args.seq_target_match_flag_specs
        if str(spec["domain"]) == str(args.seq_target_match_flag_domain))
    args.seq_d_important_side_fids = load_seq_d_important_side_fids(
        args.seq_d_important_side_fids_json)

    # Environment variables take precedence.
    args.data_dir = os.environ.get('TRAIN_DATA_PATH', args.data_dir)
    args.ckpt_dir = os.environ.get('TRAIN_CKPT_PATH', args.ckpt_dir)
    args.log_dir = os.environ.get('TRAIN_LOG_PATH', args.log_dir)
    return args


def main() -> None:
    args = parse_args()

    # Create output directories.
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    # Initialize logger and RNG.
    set_seed(args.seed)
    create_logger(os.path.join(args.log_dir, 'train.log'))
    logging.info(f"Args: {vars(args)}")
    logging.info(
        "Effective sparse reinit config: "
        f"reinit_sparse_after_epoch={args.reinit_sparse_after_epoch}, "
        f"reinit_cardinality_threshold={args.reinit_cardinality_threshold}, "
        f"num_epochs={args.num_epochs}, patience={args.patience}"
    )
    logging.info(
        "Effective seq_recent_stats config: "
        f"use_seq_recent_stats={args.use_seq_recent_stats}, "
        f"seq_recent_stats_dim={args.seq_recent_stats_dim}, "
        f"seq_recent_stats_gate_init={args.seq_recent_stats_gate_init}"
    )
    logging.info(
        "Effective pair_dense config: "
        f"use_pair_dense={args.use_pair_dense}, "
        f"pair_dense_dim={args.pair_dense_dim}, "
        f"pair_dense_gate_init={args.pair_dense_gate_init}, "
        f"pair_dense_pairs={args.pair_dense_pairs}"
    )
    logging.info(
        "Effective aligned_user_int_dense config: "
        f"use_aligned_user_int_dense={args.use_aligned_user_int_dense}, "
        f"aligned_user_int_dense_gate_init={args.aligned_user_int_dense_gate_init}, "
        f"aligned_user_int_dense_fids={args.aligned_user_int_dense_fids}"
    )
    logging.info(
        "Effective target_matched_recency config: "
        f"use_target_matched_recency={args.use_target_matched_recency}, "
        f"target_matched_recency_dim={args.target_matched_recency_dim}, "
        f"target_matched_recency_gate_init={args.target_matched_recency_gate_init}, "
        f"target_matched_recency_feature_mode={args.target_matched_recency_feature_mode}, "
        f"target_matched_recency_pairs_windows={args.target_matched_recency_pairs_windows}"
    )
    logging.info(
        "Effective seq_target_match_flags config: "
        f"use_seq_target_match_flags={args.use_seq_target_match_flags}, "
        f"seq_target_match_flag_domain={args.seq_target_match_flag_domain}, "
        f"seq_target_match_flag_num_flags={args.seq_target_match_flag_num_flags}, "
        f"seq_target_match_flag_gate_init={args.seq_target_match_flag_gate_init}, "
        f"seq_target_match_flag_specs={args.seq_target_match_flag_specs}"
    )
    logging.info(
        "Effective seq_d_side_projector config: "
        f"seq_d_side_projector_type={args.seq_d_side_projector_type}, "
        f"seq_d_important_side_fids={args.seq_d_important_side_fids}"
    )

    # ---- Data loading ----
    if args.schema_path:
        schema_path = args.schema_path
    else:
        schema_path = os.path.join(args.data_dir, 'schema.json')

    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"schema file not found at {schema_path}")

    # Parse per-domain sequence-length overrides.
    seq_max_lens = {}
    if args.seq_max_lens:
        for pair in args.seq_max_lens.split(','):
            k, v = pair.split(':')
            seq_max_lens[k.strip()] = int(v.strip())
        logging.info(f"Seq max_lens override: {seq_max_lens}")

    logging.info("Using Parquet data format (IterableDataset)")
    train_loader, valid_loader, pcvr_dataset = get_pcvr_data(
        data_dir=args.data_dir,
        schema_path=schema_path,
        batch_size=args.batch_size,
        valid_ratio=args.valid_ratio,
        valid_split_mode=args.valid_split_mode,
        valid_time_window_hours=args.valid_time_window_hours,
        train_include_valid=args.train_include_valid,
        train_ratio=args.train_ratio,
        num_workers=args.num_workers,
        buffer_batches=args.buffer_batches,
        seed=args.seed,
        seq_max_lens=seq_max_lens,
        pair_dense_pairs=args.pair_dense_pairs,
        target_matched_recency_pairs_windows=(
            args.target_matched_recency_pairs_windows
            if args.use_target_matched_recency else []),
        seq_target_match_flag_specs=(
            args.seq_target_match_flag_specs
            if args.use_seq_target_match_flags else []),
    )
    if (
        args.use_target_matched_recency
        and getattr(pcvr_dataset, "target_matched_recency_active_specs", 0) == 0
    ):
        logging.warning(
            "use_target_matched_recency=1 but no usable target-matched recency "
            "fields were found; disabling the residual to keep checkpoint shape safe")
        args.use_target_matched_recency = 0
        args.target_matched_recency_dim = 0

    # ---- NS groups ----
    if args.ns_groups_json and os.path.exists(args.ns_groups_json):
        logging.info(f"Loading NS groups from {args.ns_groups_json}")
        with open(args.ns_groups_json, 'r') as f:
            ns_groups_cfg = json.load(f)
        user_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.user_int_schema.entries)}
        item_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.item_int_schema.entries)}
        user_ns_groups = [[user_fid_to_idx[f] for f in fids] for fids in ns_groups_cfg['user_ns_groups'].values()]
        item_ns_groups = [[item_fid_to_idx[f] for f in fids] for fids in ns_groups_cfg['item_ns_groups'].values()]
        logging.info(f"User NS groups ({len(user_ns_groups)}): {list(ns_groups_cfg['user_ns_groups'].keys())}")
        logging.info(f"Item NS groups ({len(item_ns_groups)}): {list(ns_groups_cfg['item_ns_groups'].keys())}")
    else:
        logging.info("No NS groups JSON found, using default: each feature as one group")
        user_ns_groups = [[i] for i in range(len(pcvr_dataset.user_int_schema.entries))]
        item_ns_groups = [[i] for i in range(len(pcvr_dataset.item_int_schema.entries))]

    # ---- Build model ----
    user_int_feature_specs = build_feature_specs(
        pcvr_dataset.user_int_schema, pcvr_dataset.user_int_vocab_sizes)
    item_int_feature_specs = build_feature_specs(
        pcvr_dataset.item_int_schema, pcvr_dataset.item_int_vocab_sizes)

    model_args = {
        "user_int_feature_specs": user_int_feature_specs,
        "item_int_feature_specs": item_int_feature_specs,
        "user_dense_dim": pcvr_dataset.user_dense_schema.total_dim,
        "item_dense_dim": pcvr_dataset.item_dense_schema.total_dim,
        "user_int_feature_fids": pcvr_dataset.user_int_schema.feature_ids,
        "user_dense_feature_specs": pcvr_dataset.user_dense_schema.entries,
        "seq_feature_fids": pcvr_dataset.sideinfo_fids,
        "seq_vocab_sizes": pcvr_dataset.seq_domain_vocab_sizes,
        "user_ns_groups": user_ns_groups,
        "item_ns_groups": item_ns_groups,
        "d_model": args.d_model,
        "emb_dim": args.emb_dim,
        "num_queries": args.num_queries,
        "num_hyformer_blocks": args.num_hyformer_blocks,
        "num_heads": args.num_heads,
        "seq_encoder_type": args.seq_encoder_type,
        "hidden_mult": args.hidden_mult,
        "dropout_rate": args.dropout_rate,
        "seq_top_k": args.seq_top_k,
        "seq_causal": args.seq_causal,
        "action_num": args.action_num,
        "num_time_buckets": NUM_TIME_BUCKETS if args.use_time_buckets else 0,
        "rank_mixer_mode": args.rank_mixer_mode,
        "use_rope": args.use_rope,
        "rope_base": args.rope_base,
        "use_seq_time_delta_proj": args.use_seq_time_delta_proj,
        "use_time_context": bool(args.use_time_context),
        "time_context_dim": 2 + len(pcvr_dataset.seq_domains) * 3,
        "use_seq_recent_stats": bool(args.use_seq_recent_stats),
        "seq_recent_stats_dim": args.seq_recent_stats_dim,
        "seq_recent_stats_gate_init": args.seq_recent_stats_gate_init,
        "use_pair_dense": bool(args.use_pair_dense),
        "pair_dense_dim": args.pair_dense_dim,
        "pair_dense_gate_init": args.pair_dense_gate_init,
        "use_aligned_user_int_dense": bool(args.use_aligned_user_int_dense),
        "aligned_user_int_dense_gate_init": args.aligned_user_int_dense_gate_init,
        "aligned_user_int_dense_fids": args.aligned_user_int_dense_fids,
        "use_target_matched_recency": bool(args.use_target_matched_recency),
        "target_matched_recency_dim": args.target_matched_recency_dim,
        "target_matched_recency_gate_init": args.target_matched_recency_gate_init,
        "target_matched_recency_feature_mode": args.target_matched_recency_feature_mode,
        "use_seq_target_match_flags": bool(args.use_seq_target_match_flags),
        "seq_target_match_flag_num_flags": args.seq_target_match_flag_num_flags,
        "seq_target_match_flag_gate_init": args.seq_target_match_flag_gate_init,
        "seq_target_match_flag_domain": args.seq_target_match_flag_domain,
        "seq_d_side_projector_type": args.seq_d_side_projector_type,
        "seq_d_important_side_fids": args.seq_d_important_side_fids,
        "emb_skip_threshold": args.emb_skip_threshold,
        "seq_id_threshold": args.seq_id_threshold,
        "ns_tokenizer_type": args.ns_tokenizer_type,
        "user_ns_tokens": args.user_ns_tokens,
        "item_ns_tokens": args.item_ns_tokens,
        "user_dense_projector_type": args.user_dense_projector_type,
    }

    model = PCVRHyFormer(**model_args).to(args.device)
    if args.torch_compile:
        if hasattr(torch, 'compile'):
            try:
                model.forward = torch.compile(model.forward, mode=args.compile_mode)
                logging.info(f"torch.compile enabled for model.forward, mode={args.compile_mode}")
            except Exception as exc:
                logging.warning(f"torch.compile failed; continuing without compile: {exc}")
        else:
            logging.warning("torch.compile requested but this PyTorch version does not provide torch.compile")

    # Log model sizing info.
    num_sequences = len(pcvr_dataset.seq_domains)
    num_ns = model.num_ns
    T = args.num_queries * num_sequences + num_ns
    logging.info(f"PCVRHyFormer model created: num_ns={num_ns}, T={T}, d_model={args.d_model}, rank_mixer_mode={args.rank_mixer_mode}")
    logging.info(f"User NS groups: {user_ns_groups}")
    logging.info(f"Item NS groups: {item_ns_groups}")
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Total parameters: {total_params:,}")

    # ---- Training ----
    early_stopping = EarlyStopping(
        checkpoint_path=os.path.join(args.ckpt_dir, "placeholder", "model.pt"),
        patience=args.patience,
        label='model',
    )

    ckpt_params = {
        "layer": args.num_hyformer_blocks,
        "head": args.num_heads,
        "hidden": args.d_model,
    }

    trainer = PCVRHyFormerRankingTrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        lr=args.lr,
        num_epochs=args.num_epochs,
        device=args.device,
        save_dir=args.ckpt_dir,
        early_stopping=early_stopping,
        loss_type=args.loss_type,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        sparse_lr=args.sparse_lr,
        sparse_weight_decay=args.sparse_weight_decay,
        reinit_sparse_after_epoch=args.reinit_sparse_after_epoch,
        reinit_cardinality_threshold=args.reinit_cardinality_threshold,
        ckpt_params=ckpt_params,
        writer=None,
        schema_path=schema_path,
        ns_groups_path=args.ns_groups_json if args.ns_groups_json and os.path.exists(args.ns_groups_json) else None,
        eval_every_n_steps=args.eval_every_n_steps,
        train_config=vars(args),
        amp=args.amp,
        amp_dtype=args.amp_dtype,
        checkpoint_select_metric=args.checkpoint_select_metric,
        keep_top_k_checkpoints=args.keep_top_k_checkpoints,
        always_save_last_checkpoint=bool(args.always_save_last_checkpoint),
    )

    trainer.train()

    logging.info("Training complete!")


if __name__ == "__main__":
    main()
