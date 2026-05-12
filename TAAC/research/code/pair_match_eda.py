#!/usr/bin/env python3
"""EDA for target item_int x history side-info exact-match signals.

This script is intentionally read-only. It samples cloud parquet rows, checks
whether current item_int values appear in sequence side-info values, and writes
a markdown report for deciding whether a future pair_dense_feats experiment is
worth implementing.
"""

import argparse
import glob
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow.parquet as pq


@dataclass
class PairStats:
    rows: int = 0
    match_rows: int = 0
    match_count_sum: int = 0
    recent_match_rows: int = 0
    recent_match_count_sum: int = 0
    matched_last_gaps: List[float] = field(default_factory=list)
    pos_match: int = 0
    total_match_with_label: int = 0
    pos_no_match: int = 0
    total_no_match_with_label: int = 0

    def update(self, has_match: bool, match_count: int, recent_count: int,
               last_gap: Optional[float], label: Optional[int]) -> None:
        self.rows += 1
        if has_match:
            self.match_rows += 1
            self.match_count_sum += int(match_count)
            if last_gap is not None and math.isfinite(last_gap):
                self.matched_last_gaps.append(float(last_gap))
        if recent_count > 0:
            self.recent_match_rows += 1
            self.recent_match_count_sum += int(recent_count)
        if label is not None:
            if has_match:
                self.total_match_with_label += 1
                self.pos_match += int(label)
            else:
                self.total_no_match_with_label += 1
                self.pos_no_match += int(label)

    def as_row(self, item_fid: int, domain: str, side_fid: int) -> Dict[str, Any]:
        rows = max(self.rows, 1)
        match_rate = self.match_rows / rows
        recent_rate = self.recent_match_rows / rows
        pos_match_rate = (
            self.pos_match / self.total_match_with_label
            if self.total_match_with_label else math.nan
        )
        pos_no_match_rate = (
            self.pos_no_match / self.total_no_match_with_label
            if self.total_no_match_with_label else math.nan
        )
        lift = (
            pos_match_rate / pos_no_match_rate
            if pos_no_match_rate and math.isfinite(pos_match_rate) else math.nan
        )
        gaps = np.asarray(self.matched_last_gaps, dtype=np.float64)
        return {
            "item_fid": item_fid,
            "domain": domain,
            "side_fid": side_fid,
            "rows": self.rows,
            "match_any_rate": match_rate,
            "match_count_mean": self.match_count_sum / rows,
            "recent_match_any_2h_rate": recent_rate,
            "recent_match_count_2h_mean": self.recent_match_count_sum / rows,
            "matched_last_gap_mean": float(gaps.mean()) if gaps.size else math.nan,
            "matched_last_gap_p50": float(np.percentile(gaps, 50)) if gaps.size else math.nan,
            "matched_last_gap_p90": float(np.percentile(gaps, 90)) if gaps.size else math.nan,
            "positive_rate_when_match": pos_match_rate,
            "positive_rate_when_no_match": pos_no_match_rate,
            "lift": lift,
        }


def _positive_values(value: Any) -> List[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        out: List[int] = []
        for x in value:
            if x is None:
                continue
            try:
                xi = int(x)
            except (TypeError, ValueError):
                continue
            if xi > 0:
                out.append(xi)
        return out
    try:
        xi = int(value)
    except (TypeError, ValueError):
        return []
    return [xi] if xi > 0 else []


def _positive_seq_values(values: Any) -> List[int]:
    return _positive_values(values)


def _safe_label(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value) == 2
    except (TypeError, ValueError):
        return None


def _fmt(value: Any, digits: int = 6) -> str:
    if isinstance(value, float):
        if not math.isfinite(value):
            return "nan"
        return f"{value:.{digits}f}"
    return str(value)


def _markdown_table(rows: Sequence[Dict[str, Any]], cols: Sequence[str]) -> str:
    lines = ["| " + " | ".join(cols) + " |"]
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(c, "")) for c in cols) + " |")
    return "\n".join(lines)


def _iter_parquet_files(data_path: str, max_files: int) -> List[str]:
    files = sorted(glob.glob(os.path.join(data_path, "*.parquet")))
    return files[:max_files]


def _schema_candidates(schema: Dict[str, Any]) -> Tuple[List[int], Dict[str, Dict[str, Any]]]:
    item_fids = [int(fid) for fid, _vs, _dim in schema.get("item_int", [])]
    seq_info: Dict[str, Dict[str, Any]] = {}
    for domain, cfg in sorted(schema.get("seq", {}).items()):
        ts_fid = cfg.get("ts_fid")
        side_fids = [int(fid) for fid, _vs in cfg.get("features", []) if fid != ts_fid]
        seq_info[domain] = {
            "prefix": cfg["prefix"],
            "ts_fid": ts_fid,
            "side_fids": side_fids,
        }
    return item_fids, seq_info


def run_eda(args: argparse.Namespace) -> None:
    schema_path = args.schema_path or os.path.join(args.data_path, "schema.json")
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    item_fids, seq_info = _schema_candidates(schema)
    parquet_files = _iter_parquet_files(args.data_path, args.max_files)
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files under {args.data_path}")

    needed_cols = ["timestamp"]
    if args.include_label:
        needed_cols.append("label_type")
    for fid in item_fids:
        needed_cols.append(f"item_int_feats_{fid}")
    for info in seq_info.values():
        prefix = info["prefix"]
        ts_fid = info["ts_fid"]
        if ts_fid is not None:
            needed_cols.append(f"{prefix}_{ts_fid}")
        for fid in info["side_fids"]:
            needed_cols.append(f"{prefix}_{fid}")

    stats: Dict[Tuple[int, str, int], PairStats] = {}
    for item_fid in item_fids:
        for domain, info in seq_info.items():
            for side_fid in info["side_fids"]:
                stats[(item_fid, domain, side_fid)] = PairStats()

    scanned_rows = 0
    for file_path in parquet_files:
        pf = pq.ParquetFile(file_path)
        available = set(pf.schema_arrow.names)
        columns = [c for c in needed_cols if c in available]
        for batch in pf.iter_batches(batch_size=2048, columns=columns):
            n = batch.num_rows
            take_n = min(n, args.max_rows - scanned_rows)
            if take_n <= 0:
                break

            col_values = {
                name: batch.column(batch.schema.get_field_index(name)).to_pylist()[:take_n]
                for name in columns
            }
            root_ts = col_values.get("timestamp", [0] * take_n)
            labels = col_values.get("label_type")

            item_values = {
                fid: col_values.get(f"item_int_feats_{fid}", [None] * take_n)
                for fid in item_fids
            }
            for domain, info in seq_info.items():
                prefix = info["prefix"]
                ts_fid = info["ts_fid"]
                ts_col = (
                    col_values.get(f"{prefix}_{ts_fid}", [None] * take_n)
                    if ts_fid is not None else [None] * take_n
                )
                side_cols = {
                    fid: col_values.get(f"{prefix}_{fid}", [None] * take_n)
                    for fid in info["side_fids"]
                }
                for i in range(take_n):
                    try:
                        root = int(root_ts[i]) if root_ts[i] is not None else 0
                    except (TypeError, ValueError):
                        root = 0
                    seq_ts = _positive_seq_values(ts_col[i])
                    gaps = [max(root - t, 0) for t in seq_ts]
                    label = _safe_label(labels[i]) if labels is not None else None

                    for item_fid in item_fids:
                        target_values = set(_positive_values(item_values[item_fid][i]))
                        if not target_values:
                            for side_fid in info["side_fids"]:
                                stats[(item_fid, domain, side_fid)].update(
                                    False, 0, 0, None, label)
                            continue
                        for side_fid, side_values_by_row in side_cols.items():
                            seq_values = _positive_seq_values(side_values_by_row[i])
                            match_gaps: List[int] = []
                            match_count = 0
                            recent_count = 0
                            for pos, value in enumerate(seq_values):
                                if value not in target_values:
                                    continue
                                match_count += 1
                                gap = gaps[pos] if pos < len(gaps) else math.inf
                                if math.isfinite(gap):
                                    match_gaps.append(int(gap))
                                    if gap <= 7200:
                                        recent_count += 1
                            stats[(item_fid, domain, side_fid)].update(
                                match_count > 0,
                                match_count,
                                recent_count,
                                min(match_gaps) if match_gaps else None,
                                label,
                            )

            scanned_rows += take_n
            if scanned_rows >= args.max_rows:
                break
        if scanned_rows >= args.max_rows:
            break

    rows = [s.as_row(*key) for key, s in stats.items()]
    rows_by_lift = sorted(
        rows,
        key=lambda r: (
            -1 if not math.isfinite(r["lift"]) else r["lift"],
            r["match_any_rate"],
            r["recent_match_any_2h_rate"],
        ),
        reverse=True,
    )
    rows_by_match = sorted(rows, key=lambda r: r["match_any_rate"], reverse=True)
    rows_by_recent = sorted(rows, key=lambda r: r["recent_match_any_2h_rate"], reverse=True)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    cols = [
        "item_fid", "domain", "side_fid", "rows", "match_any_rate",
        "match_count_mean", "recent_match_any_2h_rate",
        "recent_match_count_2h_mean", "matched_last_gap_mean",
        "matched_last_gap_p50", "matched_last_gap_p90",
        "positive_rate_when_match", "positive_rate_when_no_match", "lift",
    ]
    promising = [
        r for r in rows_by_lift
        if math.isfinite(r["lift"]) and r["lift"] > 1.05
        and 0.001 <= r["match_any_rate"] <= 0.95
    ][:20]
    too_low = [r for r in rows_by_match if r["match_any_rate"] < 0.001][:10]
    too_high = [r for r in rows_by_match if r["match_any_rate"] > 0.95][:10]

    lines = [
        "# Pair / Target-History Match EDA",
        "",
        f"- data_path: `{args.data_path}`",
        f"- schema_path: `{schema_path}`",
        f"- parquet_files_scanned: `{len(parquet_files)}`",
        f"- rows_scanned: `{scanned_rows}`",
        "",
        "## Schema Candidates",
        "",
        f"- item_int fids: `{item_fids}`",
    ]
    for domain, info in seq_info.items():
        lines.append(
            f"- {domain}: prefix=`{info['prefix']}`, ts_fid=`{info['ts_fid']}`, "
            f"sideinfo_fids=`{info['side_fids']}`"
        )

    lines.extend([
        "",
        "## Top Pairs By Lift",
        "",
        _markdown_table(rows_by_lift[:30], cols),
        "",
        "## Top Pairs By Match Rate",
        "",
        _markdown_table(rows_by_match[:30], cols),
        "",
        "## Top Pairs By Recent 2h Match Rate",
        "",
        _markdown_table(rows_by_recent[:30], cols),
        "",
        "## Recommendations",
        "",
    ])
    if promising:
        lines.append("Pairs worth considering for future `pair_dense_feats`:")
        lines.append("")
        lines.append(_markdown_table(promising, cols))
    else:
        lines.append(
            "No strong pair met the default screen "
            "(`lift > 1.05` and `0.001 <= match_any_rate <= 0.95`)."
        )
    lines.extend([
        "",
        "Pairs with match rate too low to prioritize:",
        "",
        _markdown_table(too_low, cols) if too_low else "None.",
        "",
        "Pairs with match rate too high to be discriminative:",
        "",
        _markdown_table(too_high, cols) if too_high else "None.",
        "",
        "Interpretation:",
        "",
        "- If top lift pairs also have non-trivial recent 2h match rate, "
        "P2_pair_dense_v1 is a plausible next structural feature.",
        "- If match rates are near zero or near one, exact match likely needs "
        "a different join key or should be skipped.",
        "- This script does not train or modify model inputs.",
    ])

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {args.output}")


def parse_args() -> argparse.Namespace:
    default_data_path = os.environ.get("TRAIN_DATA_PATH", "/data_ams/academic_training_data")
    parser = argparse.ArgumentParser(description="Target-history exact-match EDA")
    parser.add_argument("--data_path", type=str, default=default_data_path)
    parser.add_argument("--schema_path", type=str, default=None)
    parser.add_argument("--max_files", type=int, default=10)
    parser.add_argument("--max_rows", type=int, default=50000)
    parser.add_argument("--output", type=str, default="research/reports/pair_match_eda.md")
    parser.add_argument("--include_label", type=int, default=1, choices=[0, 1])
    return parser.parse_args()


if __name__ == "__main__":
    run_eda(parse_args())
