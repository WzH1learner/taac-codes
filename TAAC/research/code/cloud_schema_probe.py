"""Cloud-side schema/data probe for TAAC2026 PCVR.

Run inside the training container, for example:

    TRAIN_DATA_PATH=/data_ams/academic_training_data \
    python research/code/cloud_schema_probe.py \
        --ns_groups_json train/ns_groups.json \
        --ns_tokenizer_type rankmixer --user_ns_tokens 5 --item_ns_tokens 2 \
        --num_queries 2 --d_model 64

The script samples real parquet batches and prints diagnostics only. It does
not train a model and does not infer anything from sample_data.
"""

import argparse
import json
import math
import os
from glob import glob
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq


SEQ_DOMAINS = ("seq_a", "seq_b", "seq_c", "seq_d")


class RunningStats:
    def __init__(self) -> None:
        self.n = 0
        self.sum = 0.0
        self.sumsq = 0.0
        self.min = math.inf
        self.max = -math.inf
        self.abs_max = 0.0
        self.nonzero = 0

    def update(self, arr: np.ndarray) -> None:
        x = np.asarray(arr, dtype=np.float64).ravel()
        if x.size == 0:
            return
        self.n += int(x.size)
        self.sum += float(x.sum())
        self.sumsq += float(np.square(x).sum())
        self.min = min(self.min, float(x.min()))
        self.max = max(self.max, float(x.max()))
        self.abs_max = max(self.abs_max, float(np.abs(x).max()))
        self.nonzero += int(np.count_nonzero(x))

    def as_dict(self) -> Dict[str, Any]:
        mean = self.sum / self.n if self.n else math.nan
        var = max(self.sumsq / self.n - mean * mean, 0.0) if self.n else math.nan
        return {
            "n": self.n,
            "mean": mean,
            "std": math.sqrt(var) if self.n else math.nan,
            "min": self.min if self.n else math.nan,
            "max": self.max if self.n else math.nan,
            "abs_max": self.abs_max if self.n else math.nan,
            "nonzero_rate": self.nonzero / self.n if self.n else math.nan,
        }


def _list_parquets(data_dir: str) -> List[str]:
    files = sorted(glob(os.path.join(data_dir, "*.parquet")))
    if not files:
        files = sorted(glob(os.path.join(data_dir, "**", "*.parquet"), recursive=True))
    return files


def _list_to_padded(col, max_len: Optional[int] = None, dtype=np.float32) -> np.ndarray:
    offsets = col.offsets.to_numpy()
    values = col.values.to_numpy(zero_copy_only=False)
    if max_len is None:
        max_len = max((int(offsets[i + 1] - offsets[i]) for i in range(len(offsets) - 1)), default=0)
    out = np.zeros((len(offsets) - 1, max_len), dtype=dtype)
    for i in range(len(offsets) - 1):
        start, end = int(offsets[i]), int(offsets[i + 1])
        use_len = min(end - start, max_len)
        if use_len > 0:
            out[i, :use_len] = values[start:start + use_len]
    return out


def _load_schema(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_ns_groups(schema: Dict[str, Any], ns_groups_json: str) -> Tuple[int, int]:
    if not ns_groups_json or not os.path.exists(ns_groups_json):
        return len(schema.get("user_int", [])), len(schema.get("item_int", []))
    with open(ns_groups_json, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return len(cfg["user_ns_groups"]), len(cfg["item_ns_groups"])


def _iter_batches(files: List[str], batch_size: int, max_batches: int) -> Iterable[Any]:
    seen = 0
    for path in files:
        pf = pq.ParquetFile(path)
        for rg_idx in range(pf.metadata.num_row_groups):
            for batch in pf.iter_batches(batch_size=batch_size, row_groups=[rg_idx]):
                yield batch
                seen += 1
                if seen >= max_batches:
                    return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=os.environ.get("TRAIN_DATA_PATH", "/data_ams/academic_training_data"))
    parser.add_argument("--schema_path", default=None)
    parser.add_argument("--max_batches", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--ns_groups_json", default="")
    parser.add_argument("--ns_tokenizer_type", choices=["rankmixer", "group"], default="rankmixer")
    parser.add_argument("--user_ns_tokens", type=int, default=5)
    parser.add_argument("--item_ns_tokens", type=int, default=2)
    parser.add_argument("--num_queries", type=int, default=2)
    parser.add_argument("--d_model", type=int, default=64)
    args = parser.parse_args()

    schema_path = args.schema_path or os.path.join(args.data_dir, "schema.json")
    schema = _load_schema(schema_path)
    files = _list_parquets(args.data_dir)
    print(f"DATA_DIR={args.data_dir}")
    print(f"SCHEMA={schema_path}")
    print(f"PARQUET_FILES={len(files)}")
    if files:
        pf0 = pq.ParquetFile(files[0])
        print(f"FIRST_FILE_COLUMNS={len(pf0.schema_arrow.names)}")
        print(f"FIRST_FILE_ROW_GROUPS={pf0.metadata.num_row_groups}")

    print("\n=== Schema ===")
    print(f"user_int={len(schema.get('user_int', []))}, item_int={len(schema.get('item_int', []))}, user_dense={schema.get('user_dense', [])}")
    for domain in sorted(schema.get("seq", {})):
        cfg = schema["seq"][domain]
        print(f"{domain}: prefix={cfg.get('prefix')}, ts_fid={cfg.get('ts_fid')}, features={len(cfg.get('features', []))}")

    if args.ns_tokenizer_type == "group":
        user_ns, item_ns = _resolve_ns_groups(schema, args.ns_groups_json)
    else:
        user_ns = args.user_ns_tokens or len(schema.get("user_int", []))
        item_ns = args.item_ns_tokens or len(schema.get("item_int", []))
    has_user_dense = 1 if schema.get("user_dense") else 0
    num_ns = user_ns + has_user_dense + item_ns
    T = args.num_queries * len(schema.get("seq", {})) + num_ns
    print("\n=== RankMixer T ===")
    print(f"ns_tokenizer_type={args.ns_tokenizer_type}, user_ns={user_ns}, user_dense={has_user_dense}, item_ns={item_ns}")
    print(f"T=num_queries*num_sequences+num_ns={args.num_queries}*{len(schema.get('seq', {}))}+{num_ns}={T}")
    print(f"d_model={args.d_model}, d_model%T={args.d_model % T if T else 'NA'}")

    dense_stats = {fid: RunningStats() for fid, _ in schema.get("user_dense", [])}
    label_pos = 0
    label_total = 0
    root_min = None
    root_max = None
    seq_le_root = {d: [0, 0] for d in schema.get("seq", {})}
    seq_recent_30d = {d: [0, 0] for d in schema.get("seq", {})}
    item_int_values = set()
    raw_item_values = set()

    needed_cols = {"timestamp", "label_type", "item_id"}
    for fid, _ in schema.get("user_dense", []):
        needed_cols.add(f"user_dense_feats_{fid}")
    for fid, _, _ in schema.get("item_int", []):
        needed_cols.add(f"item_int_feats_{fid}")
    for domain, cfg in schema.get("seq", {}).items():
        needed_cols.add(f"{cfg['prefix']}_{cfg['ts_fid']}")

    for batch in _iter_batches(files, args.batch_size, args.max_batches):
        names = batch.schema.names
        col_idx = {n: i for i, n in enumerate(names)}
        ts = batch.column(col_idx["timestamp"]).to_numpy(zero_copy_only=False).astype(np.int64)
        root_min = int(ts.min()) if root_min is None else min(root_min, int(ts.min()))
        root_max = int(ts.max()) if root_max is None else max(root_max, int(ts.max()))
        labels = batch.column(col_idx["label_type"]).fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
        label_pos += int((labels == 2).sum())
        label_total += int(labels.size)

        if "item_id" in col_idx:
            raw_item_values.update(batch.column(col_idx["item_id"]).fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)[:20000].tolist())
        for fid, _, dim in schema.get("item_int", []):
            name = f"item_int_feats_{fid}"
            if name not in col_idx:
                continue
            col = batch.column(col_idx[name])
            vals = col.fill_null(0).to_numpy(zero_copy_only=False) if dim == 1 else col.values.to_numpy(zero_copy_only=False)
            item_int_values.update(np.asarray(vals, dtype=np.int64)[:20000].tolist())

        for fid, dim in schema.get("user_dense", []):
            name = f"user_dense_feats_{fid}"
            if name in col_idx:
                dense_stats[fid].update(_list_to_padded(batch.column(col_idx[name]), dim, np.float32))

        for domain, cfg in schema.get("seq", {}).items():
            name = f"{cfg['prefix']}_{cfg['ts_fid']}"
            if name not in col_idx:
                continue
            seq_ts = _list_to_padded(batch.column(col_idx[name]), None, np.int64)
            valid = seq_ts > 0
            if not valid.any():
                continue
            root = ts.reshape(-1, 1)
            seq_le_root[domain][0] += int((seq_ts[valid] <= np.broadcast_to(root, seq_ts.shape)[valid]).sum())
            seq_le_root[domain][1] += int(valid.sum())
            gap = root - seq_ts
            recent = valid & (gap >= 0) & (gap <= 30 * 86400)
            seq_recent_30d[domain][0] += int(recent.sum())
            seq_recent_30d[domain][1] += int(valid.sum())

    print("\n=== Labels / Timestamp ===")
    print(f"sampled_rows={label_total}, label_type==2_rate={label_pos / max(label_total, 1):.6f}")
    print(f"root_timestamp_min={root_min}, root_timestamp_max={root_max}")
    for domain in sorted(seq_le_root):
        ok, total = seq_le_root[domain]
        recent, total_recent = seq_recent_30d[domain]
        print(
            f"{domain}: max(seq_ts)<=root_ts_rate={ok / max(total, 1):.6f}, "
            f"recent_30d_rate={recent / max(total_recent, 1):.6f}, valid_seq_ts={total}"
        )

    print("\n=== user_dense by fid ===")
    for fid in sorted(dense_stats):
        s = dense_stats[fid].as_dict()
        print(
            f"fid={fid}: dim={dict(schema.get('user_dense', [])).get(fid)}, "
            f"mean={s['mean']:.6g}, std={s['std']:.6g}, "
            f"min={s['min']:.6g}, max={s['max']:.6g}, abs_max={s['abs_max']:.6g}, "
            f"nonzero_rate={s['nonzero_rate']:.6f}"
        )

    overlap = raw_item_values.intersection(item_int_values)
    print("\n=== raw item_id vs item_int ===")
    print(f"sampled_raw_item_ids={len(raw_item_values)}, sampled_item_int_values={len(item_int_values)}, overlap={len(overlap)}")
    if overlap:
        print(f"overlap_examples={sorted(list(overlap))[:20]}")


if __name__ == "__main__":
    main()
