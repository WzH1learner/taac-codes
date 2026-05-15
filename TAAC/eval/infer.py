"""PCVRHyFormer inference script (uploaded by the contestant into the
evaluation container).

Model construction mirrors ``train.py``: we rebuild the model from
``schema.json`` + ``ns_groups.json`` + ``train_config.json``. All model
hyperparameters are resolved first from the ckpt directory's
``train_config.json`` (written by ``trainer.py`` when saving a checkpoint),
falling back to ``_FALLBACK_MODEL_CFG`` below (which must stay consistent
with the CLI defaults in ``train.py``).

Only the Parquet data format is supported.

Environment variables:
    MODEL_OUTPUT_PATH  Checkpoint directory (points at the ``global_step``
                       sub-directory containing ``model.pt`` / ``train_config.json``).
    EVAL_DATA_PATH     Test data directory (*.parquet + schema.json).
    EVAL_RESULT_PATH   Directory for the generated ``predictions.json``.
"""

import os
import json
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import FeatureSchema, PCVRParquetDataset, NUM_TIME_BUCKETS, PAIR_DENSE_DIM, SEQ_RECENT_STATS_DIM
from model import PCVRHyFormer, ModelInput


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)


# Fallback values used only when ``train_config.json`` is missing from the
# ckpt directory.
#
# These MUST match the argparse defaults in ``train.py``; otherwise once the
# fallback path is actually taken the built model will shape-mismatch the
# saved state_dict.
#
# Special note on ``num_time_buckets``: this value is strictly determined by
# ``dataset.BUCKET_BOUNDARIES`` and is NOT an independent hyperparameter.
# When the feature is enabled we therefore use the constant exposed by the
# dataset module; ``0`` means disabled.
_FALLBACK_MODEL_CFG = {
    'd_model': 64,
    'emb_dim': 64,
    'num_queries': 2,
    'num_hyformer_blocks': 2,
    'num_heads': 4,
    'seq_encoder_type': 'transformer',
    'hidden_mult': 4,
    'dropout_rate': 0.01,
    'seq_top_k': 50,
    'seq_causal': False,
    'action_num': 1,
    'num_time_buckets': NUM_TIME_BUCKETS,
    'rank_mixer_mode': 'full',
    'use_rope': False,
    'rope_base': 10000.0,
    'use_seq_time_delta_proj': False,
    'use_time_context': False,
    'time_context_dim': 14,
    'use_seq_recent_stats': False,
    'seq_recent_stats_dim': SEQ_RECENT_STATS_DIM,
    'seq_recent_stats_gate_init': 0.1,
    'use_pair_dense': False,
    'pair_dense_dim': PAIR_DENSE_DIM,
    'pair_dense_gate_init': 0.05,
    'use_aligned_user_int_dense': False,
    'aligned_user_int_dense_gate_init': 0.05,
    'aligned_user_int_dense_fids': [62, 63, 64, 65, 66, 89, 90, 91],
    'emb_skip_threshold': 5000000,
    'seq_id_threshold': 10000,
    'ns_tokenizer_type': 'rankmixer',
    'user_ns_tokens': 5,
    'item_ns_tokens': 2,
    'user_dense_projector_type': 'flat',
}

_FALLBACK_SEQ_MAX_LENS = 'seq_a:256,seq_b:256,seq_c:512,seq_d:512'
_FALLBACK_BATCH_SIZE = 256
_FALLBACK_NUM_WORKERS = 16


# Hyperparameter keys used to build the model. Everything else in
# ``train_config.json`` is ignored when constructing ``PCVRHyFormer``.
_MODEL_CFG_KEYS = list(_FALLBACK_MODEL_CFG.keys())


class EvalDistributionStats:
    """Small online diagnostics for official-test distribution drift.

    This class only reads batch tensors and logs aggregate statistics. It does
    not change model inputs, predictions, or the output JSON.
    """

    _WEEKDAY_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    _SEQ_LEN_BINS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

    def __init__(self, seq_domains: List[str]) -> None:
        self.seq_domains = list(seq_domains)
        self.num_samples = 0
        self.min_ts: Optional[int] = None
        self.max_ts: Optional[int] = None
        self.utc_hour_counts = [0] * 24
        self.cn_hour_counts = [0] * 24
        self.cn_weekday_counts = [0] * 7
        self.user_int_total = 0
        self.user_int_zero = 0
        self.item_int_total = 0
        self.item_int_zero = 0
        self.seq_len_sum = {d: 0 for d in self.seq_domains}
        self.seq_len_max = {d: 0 for d in self.seq_domains}
        self.seq_nonzero = {d: 0 for d in self.seq_domains}
        self.seq_len_bins = {d: [0] * len(self._SEQ_LEN_BINS) for d in self.seq_domains}
        self.time_bucket_total = {d: 0 for d in self.seq_domains}
        self.time_bucket_zero = {d: 0 for d in self.seq_domains}
        self.time_bucket_counts = {d: {} for d in self.seq_domains}

    @staticmethod
    def _as_cpu_long(t: torch.Tensor) -> torch.Tensor:
        return t.detach().to('cpu', dtype=torch.long)

    def update(self, batch: Dict[str, Any]) -> None:
        timestamps = self._as_cpu_long(batch['timestamp'])
        if timestamps.numel() == 0:
            return

        self.num_samples += int(timestamps.numel())
        batch_min = int(timestamps.min().item())
        batch_max = int(timestamps.max().item())
        self.min_ts = batch_min if self.min_ts is None else min(self.min_ts, batch_min)
        self.max_ts = batch_max if self.max_ts is None else max(self.max_ts, batch_max)

        utc_hours = ((timestamps // 3600) % 24).bincount(minlength=24)
        cn_hours = (((timestamps + 8 * 3600) // 3600) % 24).bincount(minlength=24)
        # 1970-01-01 is Thursday. With Monday=0, epoch day 0 maps to 3.
        cn_weekdays = (((timestamps + 8 * 3600) // 86400 + 3) % 7).bincount(minlength=7)
        for i in range(24):
            self.utc_hour_counts[i] += int(utc_hours[i].item())
            self.cn_hour_counts[i] += int(cn_hours[i].item())
        for i in range(7):
            self.cn_weekday_counts[i] += int(cn_weekdays[i].item())

        user_int = self._as_cpu_long(batch['user_int_feats'])
        item_int = self._as_cpu_long(batch['item_int_feats'])
        self.user_int_total += int(user_int.numel())
        self.user_int_zero += int((user_int == 0).sum().item())
        self.item_int_total += int(item_int.numel())
        self.item_int_zero += int((item_int == 0).sum().item())

        for domain in self.seq_domains:
            lengths = self._as_cpu_long(batch[f'{domain}_len'])
            self.seq_len_sum[domain] += int(lengths.sum().item())
            self.seq_len_max[domain] = max(self.seq_len_max[domain], int(lengths.max().item()))
            self.seq_nonzero[domain] += int((lengths > 0).sum().item())
            for j, upper in enumerate(self._SEQ_LEN_BINS):
                if j == 0:
                    count = int((lengths == 0).sum().item())
                else:
                    prev = self._SEQ_LEN_BINS[j - 1]
                    count = int(((lengths > prev) & (lengths <= upper)).sum().item())
                self.seq_len_bins[domain][j] += count

            tb = self._as_cpu_long(batch[f'{domain}_time_bucket'])
            self.time_bucket_total[domain] += int(tb.numel())
            self.time_bucket_zero[domain] += int((tb == 0).sum().item())
            flat_nonzero = tb[tb > 0]
            if flat_nonzero.numel() > 0:
                vals, counts = torch.unique(flat_nonzero, return_counts=True)
                bucket_counts = self.time_bucket_counts[domain]
                for value, count in zip(vals.tolist(), counts.tolist()):
                    bucket_counts[int(value)] = bucket_counts.get(int(value), 0) + int(count)

    @staticmethod
    def _top_counts(counts: List[int], labels: Optional[List[str]] = None, top_k: int = 8) -> str:
        pairs = [(i, c) for i, c in enumerate(counts) if c > 0]
        pairs.sort(key=lambda x: x[1], reverse=True)
        parts = []
        total = sum(counts)
        for i, c in pairs[:top_k]:
            label = labels[i] if labels else str(i)
            pct = 100.0 * c / max(total, 1)
            parts.append(f"{label}:{c}({pct:.2f}%)")
        return ', '.join(parts) if parts else 'N/A'

    def log(self) -> None:
        logging.info("Eval data diagnostics: samples=%d, timestamp_min=%s, timestamp_max=%s",
                     self.num_samples, self.min_ts, self.max_ts)
        if self.min_ts is not None and self.max_ts is not None:
            span_hours = (self.max_ts - self.min_ts) / 3600.0
            logging.info("Eval data diagnostics: timestamp_span_hours=%.2f", span_hours)
        logging.info(
            "Eval data diagnostics: top UTC hours=%s",
            self._top_counts(self.utc_hour_counts),
        )
        logging.info(
            "Eval data diagnostics: top CN hours=%s",
            self._top_counts(self.cn_hour_counts),
        )
        logging.info(
            "Eval data diagnostics: top CN weekdays=%s",
            self._top_counts(self.cn_weekday_counts, self._WEEKDAY_NAMES),
        )

        def zero_rate(zero: int, total: int) -> float:
            return zero / total if total else math.nan

        logging.info(
            "Eval data diagnostics: user_int_zero_rate=%.6f, item_int_zero_rate=%.6f",
            zero_rate(self.user_int_zero, self.user_int_total),
            zero_rate(self.item_int_zero, self.item_int_total),
        )

        for domain in self.seq_domains:
            avg_len = self.seq_len_sum[domain] / max(self.num_samples, 1)
            nonzero_rate = self.seq_nonzero[domain] / max(self.num_samples, 1)
            len_parts = []
            for upper, count in zip(self._SEQ_LEN_BINS, self.seq_len_bins[domain]):
                if count:
                    label = '0' if upper == 0 else f"<={upper}"
                    len_parts.append(f"{label}:{count}")
            tb_total = self.time_bucket_total[domain]
            tb_zero_rate = self.time_bucket_zero[domain] / max(tb_total, 1)
            tb_counts = self.time_bucket_counts[domain]
            top_tb = sorted(tb_counts.items(), key=lambda x: x[1], reverse=True)[:8]
            top_tb_str = ', '.join(
                f"{bucket}:{count}({100.0 * count / max(tb_total, 1):.2f}%)"
                for bucket, count in top_tb
            ) or 'N/A'
            logging.info(
                "Eval data diagnostics: %s len_avg=%.3f, len_max=%d, nonzero_rate=%.6f, len_bins=%s",
                domain, avg_len, self.seq_len_max[domain], nonzero_rate,
                ', '.join(len_parts) or 'N/A',
            )
            logging.info(
                "Eval data diagnostics: %s time_bucket_zero_rate=%.6f, top_nonzero_buckets=%s",
                domain, tb_zero_rate, top_tb_str,
            )


def build_feature_specs(
    schema: FeatureSchema,
    per_position_vocab_sizes: List[int],
) -> List[Tuple[int, int, int]]:
    """Build ``feature_specs = [(vocab_size, offset, length), ...]`` in the
    order of ``schema.entries``.
    """
    specs: List[Tuple[int, int, int]] = []
    for fid, offset, length in schema.entries:
        vs = max(per_position_vocab_sizes[offset:offset + length])
        specs.append((vs, offset, length))
    return specs


def _parse_seq_max_lens(sml_str: str) -> Dict[str, int]:
    """Parse a string like ``'seq_a:256,seq_b:256,...'`` into a dict."""
    seq_max_lens: Dict[str, int] = {}
    for pair in sml_str.split(','):
        k, v = pair.split(':')
        seq_max_lens[k.strip()] = int(v.strip())
    return seq_max_lens


def load_train_config(model_dir: str) -> Dict[str, Any]:
    """Load ``train_config.json`` from the ckpt directory.

    Returns an empty dict (which triggers fallback resolution) if the file is
    not present.
    """
    train_config_path = os.path.join(model_dir, 'train_config.json')
    if os.path.exists(train_config_path):
        with open(train_config_path, 'r') as f:
            cfg = json.load(f)
        logging.info(f"Loaded train_config from {train_config_path}")
        return cfg
    logging.warning(
        f"train_config.json not found in {model_dir}, "
        f"falling back to hardcoded defaults. "
        f"Shape mismatch may occur if training used non-default hyperparameters.")
    return {}


def resolve_model_cfg(train_config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract model hyperparameters from ``train_config``; missing keys fall
    back to ``_FALLBACK_MODEL_CFG``.

    Special handling for ``num_time_buckets``: it is not exposed on the CLI
    as an independent hyperparameter; the bucket count is uniquely determined
    by the length of ``dataset.BUCKET_BOUNDARIES``. Resolution order:

      1) ``train_config`` contains ``num_time_buckets`` directly (legacy ckpt)
         -> use that value;
      2) ``train_config`` contains ``use_time_buckets`` (new-style training)
         -> derive as ``NUM_TIME_BUCKETS`` or ``0``;
      3) neither is present -> fall back to ``_FALLBACK_MODEL_CFG[...]``.
    """
    cfg: Dict[str, Any] = {}
    for key in _MODEL_CFG_KEYS:
        if key == 'num_time_buckets':
            if 'num_time_buckets' in train_config:
                cfg[key] = train_config['num_time_buckets']
            elif 'use_time_buckets' in train_config:
                cfg[key] = NUM_TIME_BUCKETS if train_config['use_time_buckets'] else 0
            else:
                cfg[key] = _FALLBACK_MODEL_CFG[key]
                logging.warning(
                    f"train_config missing both 'num_time_buckets' and 'use_time_buckets', "
                    f"using fallback = {cfg[key]}")
            continue

        if key in train_config:
            cfg[key] = train_config[key]
        else:
            cfg[key] = _FALLBACK_MODEL_CFG[key]
            logging.warning(
                f"train_config missing '{key}', using fallback = {cfg[key]}")
    return cfg


def build_model(
    dataset: PCVRParquetDataset,
    model_cfg: Dict[str, Any],
    ns_groups_json: Optional[str] = None,
    device: str = 'cpu',
) -> PCVRHyFormer:
    """Construct a ``PCVRHyFormer`` from the dataset schema, an NS-groups JSON,
    and a resolved ``model_cfg`` dict.

    Args:
        dataset: a ``PCVRParquetDataset`` providing the feature schema.
        model_cfg: resolved model hyperparameters, typically the output of
            ``resolve_model_cfg``.
        ns_groups_json: path to the NS-groups JSON file, or ``None`` / empty
            string to disable it (each feature becomes its own singleton group).
        device: torch device.
    """
    # NS grouping. The JSON schema uses *fid* (feature id) values; convert
    # them to positional indices into ``user_int_schema.entries`` /
    # ``item_int_schema.entries`` so ``GroupNSTokenizer`` /
    # ``RankMixerNSTokenizer`` can index ``feature_specs`` directly. This is
    # the same conversion ``train.py`` performs when loading the JSON; doing
    # it here keeps infer.py symmetric with training.
    user_ns_groups: List[List[int]]
    item_ns_groups: List[List[int]]
    if ns_groups_json and os.path.exists(ns_groups_json):
        logging.info(f"Loading NS groups from {ns_groups_json}")
        with open(ns_groups_json, 'r') as f:
            ns_groups_cfg = json.load(f)
        user_fid_to_idx = {
            fid: i for i, (fid, _, _) in enumerate(dataset.user_int_schema.entries)
        }
        item_fid_to_idx = {
            fid: i for i, (fid, _, _) in enumerate(dataset.item_int_schema.entries)
        }
        try:
            user_ns_groups = [
                [user_fid_to_idx[f] for f in fids]
                for fids in ns_groups_cfg['user_ns_groups'].values()
            ]
            item_ns_groups = [
                [item_fid_to_idx[f] for f in fids]
                for fids in ns_groups_cfg['item_ns_groups'].values()
            ]
        except KeyError as exc:
            raise KeyError(
                f"NS-groups JSON references fid {exc.args[0]} which is not "
                f"present in the checkpoint's schema.json. The ns_groups.json "
                f"and schema.json must come from the same training run."
            ) from exc
    else:
        logging.info("No NS groups JSON found, using default: each feature as one group")
        user_ns_groups = [[i] for i in range(len(dataset.user_int_schema.entries))]
        item_ns_groups = [[i] for i in range(len(dataset.item_int_schema.entries))]

    # Feature specs.
    user_int_feature_specs = build_feature_specs(
        dataset.user_int_schema, dataset.user_int_vocab_sizes)
    item_int_feature_specs = build_feature_specs(
        dataset.item_int_schema, dataset.item_int_vocab_sizes)

    logging.info(f"Building PCVRHyFormer with cfg: {model_cfg}")
    model = PCVRHyFormer(
        user_int_feature_specs=user_int_feature_specs,
        item_int_feature_specs=item_int_feature_specs,
        user_dense_dim=dataset.user_dense_schema.total_dim,
        item_dense_dim=dataset.item_dense_schema.total_dim,
        user_int_feature_fids=dataset.user_int_schema.feature_ids,
        user_dense_feature_specs=dataset.user_dense_schema.entries,
        seq_vocab_sizes=dataset.seq_domain_vocab_sizes,
        user_ns_groups=user_ns_groups,
        item_ns_groups=item_ns_groups,
        **model_cfg,
    ).to(device)

    return model


def load_model_state_strict(
    model: nn.Module,
    ckpt_path: str,
    device: str,
) -> None:
    """Strictly load ``state_dict``; any missing/unexpected key fails fast
    with a diagnostic message.
    """
    state_dict = torch.load(ckpt_path, map_location=device)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        logging.error(
            "Failed to load state_dict in strict mode. This usually means the "
            "model constructed by build_model does NOT match the checkpoint. "
            "Check that train_config.json in the ckpt dir is present and matches "
            "the training hyperparameters.")
        raise e


def get_ckpt_path() -> Optional[str]:
    """Locate the first ``*.pt`` file inside the directory pointed at by
    ``$MODEL_OUTPUT_PATH``. Returns ``None`` if no checkpoint is found.
    """
    ckpt_path = os.environ.get("MODEL_OUTPUT_PATH")
    if not ckpt_path:
        return None
    for item in os.listdir(ckpt_path):
        if item.endswith(".pt"):
            return os.path.join(ckpt_path, item)
    return None


def _batch_to_model_input(
    batch: Dict[str, Any],
    device: str,
) -> ModelInput:
    """Convert a batch dict to ``ModelInput``, handling dynamic seq domains."""
    device_batch: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            device_batch[k] = v.to(device, non_blocking=True)
        else:
            device_batch[k] = v

    seq_domains = device_batch['_seq_domains']
    seq_data: Dict[str, torch.Tensor] = {}
    seq_lens: Dict[str, torch.Tensor] = {}
    seq_time_buckets: Dict[str, torch.Tensor] = {}
    seq_time_deltas: Dict[str, torch.Tensor] = {}
    for domain in seq_domains:
        seq_data[domain] = device_batch[domain]
        seq_lens[domain] = device_batch[f'{domain}_len']
        B, _, L = device_batch[domain].shape
        seq_time_buckets[domain] = device_batch.get(
            f'{domain}_time_bucket',
            torch.zeros(B, L, dtype=torch.long, device=device))
        seq_time_deltas[domain] = device_batch.get(
            f'{domain}_time_delta',
            torch.zeros(B, L, dtype=torch.float32, device=device))

    return ModelInput(
        user_int_feats=device_batch['user_int_feats'],
        item_int_feats=device_batch['item_int_feats'],
        user_dense_feats=device_batch['user_dense_feats'],
        item_dense_feats=device_batch['item_dense_feats'],
        seq_data=seq_data,
        seq_lens=seq_lens,
        seq_time_buckets=seq_time_buckets,
        seq_time_deltas=seq_time_deltas,
        time_context=device_batch.get('time_context', None),
        seq_recent_stats=device_batch.get('seq_recent_stats', None),
        pair_dense_feats=device_batch.get('pair_dense_feats', None),
    )


def main() -> None:
    # ---- Read environment variables ----
    model_dir = os.environ.get('MODEL_OUTPUT_PATH')
    data_dir = os.environ.get('EVAL_DATA_PATH')
    result_dir = os.environ.get('EVAL_RESULT_PATH')

    os.makedirs(result_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ---- Schema: prefer the one from model_dir (to exactly match training);
    #      fall back to the one in data_dir if missing. ----
    schema_path = os.path.join(model_dir, 'schema.json')
    if not os.path.exists(schema_path):
        schema_path = os.path.join(data_dir, 'schema.json')
    logging.info(f"Using schema: {schema_path}")

    # ---- Load train_config.json (single source of truth for all hyperparams) ----
    train_config = load_train_config(model_dir)

    # ---- Parse seq_max_lens ----
    sml_str = train_config.get('seq_max_lens', _FALLBACK_SEQ_MAX_LENS)
    seq_max_lens = _parse_seq_max_lens(sml_str)
    logging.info(f"seq_max_lens: {seq_max_lens}")

    # ---- Data loading: reuse batch_size / num_workers from training config ----
    batch_size = int(train_config.get('batch_size', _FALLBACK_BATCH_SIZE))
    num_workers = int(train_config.get('num_workers', _FALLBACK_NUM_WORKERS))

    test_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=False,
        buffer_batches=0,
        is_training=False,
        pair_dense_pairs=train_config.get('pair_dense_pairs', None),
    )
    total_test_samples = test_dataset.num_rows
    logging.info(f"Total test samples: {total_test_samples}")

    # ---- Build model: every structural hyperparameter is resolved from train_config ----
    model_cfg = resolve_model_cfg(train_config)
    logging.info(
        "Effective seq_recent_stats config: use_seq_recent_stats=%s, "
        "seq_recent_stats_dim=%s, seq_recent_stats_gate_init=%s",
        model_cfg.get('use_seq_recent_stats'),
        model_cfg.get('seq_recent_stats_dim'),
        model_cfg.get('seq_recent_stats_gate_init'),
    )
    logging.info(
        "Effective pair_dense config: use_pair_dense=%s, "
        "pair_dense_dim=%s, pair_dense_gate_init=%s",
        model_cfg.get('use_pair_dense'),
        model_cfg.get('pair_dense_dim'),
        model_cfg.get('pair_dense_gate_init'),
    )
    logging.info(
        "Effective aligned_user_int_dense config: use_aligned_user_int_dense=%s, "
        "aligned_user_int_dense_gate_init=%s, aligned_user_int_dense_fids=%s",
        model_cfg.get('use_aligned_user_int_dense'),
        model_cfg.get('aligned_user_int_dense_gate_init'),
        model_cfg.get('aligned_user_int_dense_fids'),
    )

    # ns_groups_json also comes from training config (e.g. run.sh may have
    # passed an empty string to disable it). When trainer.py has copied the
    # JSON into the ckpt dir, train_config records just the basename, so try
    # resolving against ``model_dir`` first before honoring the raw (possibly
    # absolute) path as a fallback.
    ns_groups_json = train_config.get('ns_groups_json', None)
    if ns_groups_json:
        local_candidate = os.path.join(model_dir, os.path.basename(ns_groups_json))
        if os.path.exists(local_candidate):
            ns_groups_json = local_candidate

    model = build_model(
        test_dataset,
        model_cfg=model_cfg,
        ns_groups_json=ns_groups_json,
        device=device,
    )

    # ---- Strictly load weights ----
    ckpt_path = get_ckpt_path()
    if ckpt_path is None:
        raise FileNotFoundError(
            f"No *.pt file found under MODEL_OUTPUT_PATH={model_dir!r}. "
            f"The directory contains: {os.listdir(model_dir) if model_dir and os.path.isdir(model_dir) else 'N/A'}. "
            "This typically means the training job wrote only the sidecar "
            "files (schema.json / train_config.json) for this step but did "
            "not persist model.pt — a symptom of a race between "
            "_remove_old_best_dirs and EarlyStopping.save_checkpoint."
        )
    logging.info(f"Loading checkpoint from {ckpt_path}")
    load_model_state_strict(model, ckpt_path, device)
    model.eval()
    logging.info("Model loaded successfully")

    test_loader = DataLoader(
        test_dataset,
        batch_size=None,
        num_workers=num_workers,
        prefetch_factor=2,
        pin_memory=torch.cuda.is_available(),
    )

    all_probs = []
    all_user_ids = []
    eval_stats = EvalDistributionStats(test_dataset.seq_domains)
    logging.info("Starting inference...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            eval_stats.update(batch)
            model_input = _batch_to_model_input(batch, device)
            user_ids = batch.get('user_id', [])

            logits, _ = model.predict(model_input)
            logits = logits.squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_user_ids.extend(user_ids)

            if (batch_idx + 1) % 100 == 0:
                logging.info(f"  Processed {(batch_idx + 1) * batch_size} samples")

    logging.info(f"Inference complete: {len(all_probs)} predictions")
    eval_stats.log()

    predictions = {
        "predictions": dict(zip(all_user_ids, all_probs)),
    }

    # ---- Save predictions.json ----
    output_path = os.path.join(result_dir, 'predictions.json')
    with open(output_path, 'w') as f:
        json.dump(predictions, f)
    logging.info(f"Saved {len(all_probs)} predictions to {output_path}")


if __name__ == "__main__":
    main()
