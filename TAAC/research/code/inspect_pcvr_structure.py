"""Inspect TAAC PCVR data/schema structure for model-configuration decisions.

This is a read-only EDA helper. It works with just a Parquet sample, and gives
extra model-facing diagnostics when a platform schema.json is supplied.

Examples:
    python research/code/inspect_pcvr_structure.py \
        --parquet sample_data/demo_1000.parquet

    python research/code/inspect_pcvr_structure.py \
        --parquet /path/to/train_data \
        --schema /path/to/schema.json \
        --ns-groups train/ns_groups.json \
        --emb-skip-threshold 1000000 \
        --user-ns-tokens 5 \
        --item-ns-tokens 2
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import pyarrow.parquet as pq


def _is_list_like(value: Any) -> bool:
    return hasattr(value, "__len__") and not isinstance(value, (str, bytes))


def _fid_from_col(name: str) -> int:
    return int(name.rsplit("_", 1)[-1])


def _first_parquet(path: str) -> str:
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.parquet")))
        if not files:
            raise FileNotFoundError(f"No parquet files found under {path}")
        return files[0]
    return path


def inspect_parquet(path: str, max_rows: int) -> None:
    parquet_path = _first_parquet(path)
    pf = pq.ParquetFile(parquet_path)
    cols = pf.schema_arrow.names
    print("== Parquet ==")
    print(f"path={parquet_path}")
    print(f"rows={pf.metadata.num_rows}, row_groups={pf.metadata.num_row_groups}, cols={len(cols)}")

    df = pd.read_parquet(parquet_path)
    if max_rows > 0:
        df = df.head(max_rows)
    print(f"loaded_rows_for_stats={len(df)}")

    if "label_type" in df.columns:
        counts = Counter(df["label_type"].fillna(-1).astype(int).tolist())
        print(f"label_type_counts={dict(sorted(counts.items()))}")
        print(f"label_type_2_rate={float((df['label_type'] == 2).mean()):.6f}")

    categories = {
        "id_label": [c for c in cols if c in {"user_id", "item_id", "label_type", "label_time", "timestamp"}],
        "user_int": [c for c in cols if c.startswith("user_int_feats_")],
        "user_dense": [c for c in cols if c.startswith("user_dense_feats_")],
        "item_int": [c for c in cols if c.startswith("item_int_feats_")],
        "seq_a": [c for c in cols if c.startswith("domain_a_seq_")],
        "seq_b": [c for c in cols if c.startswith("domain_b_seq_")],
        "seq_c": [c for c in cols if c.startswith("domain_c_seq_")],
        "seq_d": [c for c in cols if c.startswith("domain_d_seq_")],
    }
    for name, values in categories.items():
        print(f"{name}: n={len(values)}, cols={values[:8]}{'...' if len(values) > 8 else ''}")

    print("\n== Sequence Lengths From Parquet Lists ==")
    for domain in ("seq_a", "seq_b", "seq_c", "seq_d"):
        seq_cols = categories[domain]
        if not seq_cols:
            continue
        c = seq_cols[0]
        lens = df[c].map(lambda x: len(x) if _is_list_like(x) else 0)
        print(
            f"{domain}: representative_col={c}, "
            f"mean={lens.mean():.2f}, p50={int(lens.quantile(.50))}, "
            f"p90={int(lens.quantile(.90))}, max={int(lens.max())}, "
            f"empty={int((lens == 0).sum())}/{len(lens)}"
        )

    print("\n== First Row Example ==")
    row = df.iloc[0]
    for c in ["user_id", "item_id", "label_type", "timestamp"]:
        if c in df.columns:
            print(f"{c}={row[c]}")
    for c in [
        "user_int_feats_1", "user_int_feats_15", "user_dense_feats_61",
        "item_int_feats_5", "item_int_feats_11",
        "domain_a_seq_38", "domain_b_seq_67", "domain_c_seq_27", "domain_d_seq_17",
    ]:
        if c not in df.columns:
            continue
        value = row[c]
        if _is_list_like(value):
            print(f"{c}: len={len(value)}, head={list(value)[:8]}")
        else:
            print(f"{c}: {value}")


def _load_ns_groups(
    schema_entries: Sequence[Sequence[int]],
    ns_groups_path: Optional[str],
    group_key: str,
) -> List[List[int]]:
    if not ns_groups_path:
        return [[i] for i in range(len(schema_entries))]

    with open(ns_groups_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    fid_to_idx = {int(fid): i for i, (fid, _, _) in enumerate(schema_entries)}
    groups = []
    for fids in raw[group_key].values():
        groups.append([fid_to_idx[int(fid)] for fid in fids if int(fid) in fid_to_idx])
    return groups


def _rankmixer_chunks(
    entries: Sequence[Sequence[int]],
    groups: Sequence[Sequence[int]],
    num_tokens: int,
    emb_dim: int,
) -> None:
    flat = [idx for group in groups for idx in group]
    total_emb_dim = len(flat) * emb_dim
    chunk_dim = (total_emb_dim + num_tokens - 1) // num_tokens
    padded_total_dim = chunk_dim * num_tokens
    print(
        f"num_fids={len(flat)}, num_tokens={num_tokens}, emb_dim={emb_dim}, "
        f"total_emb_dim={total_emb_dim}, chunk_dim={chunk_dim}, pad={padded_total_dim - total_emb_dim}"
    )
    if chunk_dim % emb_dim != 0:
        print(
            f"WARNING: chunk_dim={chunk_dim} is not divisible by emb_dim={emb_dim}; "
            "RankMixerNSTokenizer chunks split inside at least one feature embedding."
        )

    for token_idx in range(num_tokens):
        start = token_idx * chunk_dim
        end = min((token_idx + 1) * chunk_dim, total_emb_dim)
        first_fid = start // emb_dim
        last_fid = (max(end - 1, start)) // emb_dim
        fids = [entries[flat[i]][0] for i in range(first_fid, min(last_fid + 1, len(flat)))]
        boundary = "" if start % emb_dim == 0 and end % emb_dim == 0 else " (partial-boundary)"
        print(f"  token{token_idx}: dim[{start}:{end}]{boundary}, fids={fids}")


def inspect_schema(
    schema_path: str,
    ns_groups_path: Optional[str],
    emb_skip_threshold: int,
    user_ns_tokens: int,
    item_ns_tokens: int,
    emb_dim: int,
) -> None:
    with open(schema_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    print("\n== schema.json ==")
    for key in ("user_int", "item_int", "user_dense"):
        print(f"{key}: n={len(raw.get(key, []))}")
    print(f"seq_domains={list(sorted(raw.get('seq', {}).keys()))}")

    for name, entries in [("user_int", raw["user_int"]), ("item_int", raw["item_int"])]:
        skipped = [(fid, vs, dim) for fid, vs, dim in entries if emb_skip_threshold > 0 and int(vs) > emb_skip_threshold]
        print(f"{name} skipped_by_emb_skip_threshold={len(skipped)}/{len(entries)}: {skipped}")

    for domain, cfg in sorted(raw.get("seq", {}).items()):
        sideinfo = [(fid, vs) for fid, vs in cfg["features"] if fid != cfg.get("ts_fid")]
        skipped = [(fid, vs) for fid, vs in sideinfo if emb_skip_threshold > 0 and int(vs) > emb_skip_threshold]
        print(
            f"{domain}: prefix={cfg.get('prefix')}, ts_fid={cfg.get('ts_fid')}, "
            f"sideinfo={len(sideinfo)}, skipped={len(skipped)}/{len(sideinfo)} {skipped}"
        )

    print("\n== RankMixer NS Token Chunking ==")
    user_groups = _load_ns_groups(raw["user_int"], ns_groups_path, "user_ns_groups")
    item_groups = _load_ns_groups(raw["item_int"], ns_groups_path, "item_ns_groups")
    print(f"user groups={len(user_groups)} ({'ns_groups.json' if ns_groups_path else 'singleton/default'})")
    _rankmixer_chunks(raw["user_int"], user_groups, user_ns_tokens, emb_dim)
    print(f"item groups={len(item_groups)} ({'ns_groups.json' if ns_groups_path else 'singleton/default'})")
    _rankmixer_chunks(raw["item_int"], item_groups, item_ns_tokens, emb_dim)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True, help="Parquet file or directory")
    parser.add_argument("--schema", default="", help="Optional platform schema.json")
    parser.add_argument("--ns-groups", default="", help="Optional ns_groups.json")
    parser.add_argument("--max-rows", type=int, default=1000)
    parser.add_argument("--emb-skip-threshold", type=int, default=1000000)
    parser.add_argument("--user-ns-tokens", type=int, default=5)
    parser.add_argument("--item-ns-tokens", type=int, default=2)
    parser.add_argument("--emb-dim", type=int, default=64)
    args = parser.parse_args()

    inspect_parquet(args.parquet, args.max_rows)
    if args.schema:
        inspect_schema(
            schema_path=args.schema,
            ns_groups_path=args.ns_groups or None,
            emb_skip_threshold=args.emb_skip_threshold,
            user_ns_tokens=args.user_ns_tokens,
            item_ns_tokens=args.item_ns_tokens,
            emb_dim=args.emb_dim,
        )


if __name__ == "__main__":
    main()
