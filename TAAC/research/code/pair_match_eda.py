#!/usr/bin/env python3
"""EDA for target item_int x history side-info exact-match signals.

The script is read-only: it samples cloud parquet rows, checks whether current
item_int values appear in sequence side-info values, and writes a markdown
report for deciding whether a future pair_dense_feats experiment is stable
enough to train.
"""

import argparse
import glob
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

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

    def update(
        self,
        has_match: bool,
        match_count: int,
        recent_count: int,
        last_gap: Optional[float],
        label: Optional[int],
    ) -> None:
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
            "match_count_abs": self.match_rows,
            "recent_match_count_abs_2h": self.recent_match_rows,
            "match_any_rate": self.match_rows / rows,
            "match_count_mean": self.match_count_sum / rows,
            "recent_match_any_2h_rate": self.recent_match_rows / rows,
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


def _safe_label(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(int(value) == 2)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any, digits: int = 6) -> str:
    if isinstance(value, float):
        if not math.isfinite(value):
            return "nan"
        return f"{value:.{digits}f}"
    return str(value)


def _markdown_table(rows: Sequence[Dict[str, Any]], cols: Sequence[str]) -> str:
    if not rows:
        return "None."
    lines = ["| " + " | ".join(cols) + " |"]
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(c, "")) for c in cols) + " |")
    return "\n".join(lines)


def _iter_parquet_files(data_path: str, max_files: int) -> List[str]:
    return sorted(glob.glob(os.path.join(data_path, "*.parquet")))[:max_files]


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


def _empty_stats(item_fids: Sequence[int], seq_info: Dict[str, Dict[str, Any]]) -> Dict[Tuple[int, str, int], PairStats]:
    stats: Dict[Tuple[int, str, int], PairStats] = {}
    for item_fid in item_fids:
        for domain, info in seq_info.items():
            for side_fid in info["side_fids"]:
                stats[(item_fid, domain, side_fid)] = PairStats()
    return stats


def _scan_rows(
    data_path: str,
    parquet_files: Sequence[str],
    item_fids: Sequence[int],
    seq_info: Dict[str, Dict[str, Any]],
    max_rows: int,
    include_label: bool,
) -> List[Dict[str, Any]]:
    needed_cols = ["timestamp"]
    if include_label:
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

    rows: List[Dict[str, Any]] = []
    for file_path in parquet_files:
        pf = pq.ParquetFile(file_path)
        available = set(pf.schema_arrow.names)
        columns = [c for c in needed_cols if c in available]
        for batch in pf.iter_batches(batch_size=2048, columns=columns):
            remaining = max_rows - len(rows)
            if remaining <= 0:
                break
            take_n = min(batch.num_rows, remaining)
            col_values = {
                name: batch.column(batch.schema.get_field_index(name)).to_pylist()[:take_n]
                for name in columns
            }
            for i in range(take_n):
                row: Dict[str, Any] = {
                    "timestamp": col_values.get("timestamp", [0] * take_n)[i],
                    "label": _safe_label(col_values.get("label_type", [None] * take_n)[i])
                    if include_label else None,
                    "item": {},
                    "seq": {},
                }
                for fid in item_fids:
                    row["item"][fid] = col_values.get(f"item_int_feats_{fid}", [None] * take_n)[i]
                for domain, info in seq_info.items():
                    prefix = info["prefix"]
                    ts_fid = info["ts_fid"]
                    domain_row = {"ts": None, "side": {}}
                    if ts_fid is not None:
                        domain_row["ts"] = col_values.get(f"{prefix}_{ts_fid}", [None] * take_n)[i]
                    for side_fid in info["side_fids"]:
                        domain_row["side"][side_fid] = col_values.get(
                            f"{prefix}_{side_fid}", [None] * take_n)[i]
                    row["seq"][domain] = domain_row
                rows.append(row)
            if len(rows) >= max_rows:
                break
        if len(rows) >= max_rows:
            break
    return rows


def _accumulate_rows(
    rows: Sequence[Dict[str, Any]],
    item_fids: Sequence[int],
    seq_info: Dict[str, Dict[str, Any]],
) -> Dict[Tuple[int, str, int], PairStats]:
    stats = _empty_stats(item_fids, seq_info)
    for row in rows:
        try:
            root = int(row["timestamp"]) if row["timestamp"] is not None else 0
        except (TypeError, ValueError):
            root = 0
        label = row["label"]
        for domain, info in seq_info.items():
            seq_row = row["seq"][domain]
            seq_ts = _positive_values(seq_row["ts"])
            gaps = [max(root - t, 0) for t in seq_ts]
            for item_fid in item_fids:
                target_values = set(_positive_values(row["item"][item_fid]))
                if not target_values:
                    for side_fid in info["side_fids"]:
                        stats[(item_fid, domain, side_fid)].update(False, 0, 0, None, label)
                    continue
                for side_fid in info["side_fids"]:
                    seq_values = _positive_values(seq_row["side"][side_fid])
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
    return stats


def _rows_from_stats(stats: Dict[Tuple[int, str, int], PairStats]) -> List[Dict[str, Any]]:
    return [s.as_row(*key) for key, s in stats.items()]


def _sort_by_lift(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        rows,
        key=lambda r: (
            math.isfinite(r["lift"]),
            -math.inf if not math.isfinite(r["lift"]) else r["lift"],
            r["match_any_rate"],
            r["recent_match_any_2h_rate"],
        ),
        reverse=True,
    )


def _stable_candidates(rows: Sequence[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    return [
        r for r in rows
        if r["match_count_abs"] >= args.min_match_count
        and r["recent_match_count_abs_2h"] >= args.min_recent_match_count
        and r["match_any_rate"] >= 0.005
        and math.isfinite(r["lift"])
        and r["lift"] > 1.05
        and r["positive_rate_when_match"] > r["positive_rate_when_no_match"]
    ]


def _unstable_candidates(rows: Sequence[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    return [
        r for r in rows
        if math.isfinite(r["lift"])
        and r["lift"] > 1.20
        and (r["match_count_abs"] < args.min_match_count
             or r["recent_match_count_abs_2h"] < args.min_recent_match_count)
    ]


def run_eda(args: argparse.Namespace) -> None:
    schema_path = args.schema_path or os.path.join(args.data_path, "schema.json")
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    item_fids, seq_info = _schema_candidates(schema)
    parquet_files = _iter_parquet_files(args.data_path, args.max_files)
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files under {args.data_path}")

    rows_raw = _scan_rows(
        data_path=args.data_path,
        parquet_files=parquet_files,
        item_fids=item_fids,
        seq_info=seq_info,
        max_rows=args.max_rows,
        include_label=bool(args.include_label),
    )
    stats = _accumulate_rows(rows_raw, item_fids, seq_info)
    rows = _rows_from_stats(stats)
    rows_by_lift = _sort_by_lift(rows)
    rows_by_match = sorted(rows, key=lambda r: r["match_any_rate"], reverse=True)
    rows_by_recent = sorted(rows, key=lambda r: r["recent_match_any_2h_rate"], reverse=True)
    stable = _sort_by_lift(_stable_candidates(rows, args))
    unstable = _sort_by_lift(_unstable_candidates(rows, args))
    focus = [
        r for r in rows
        if r["domain"] == args.focus_domain and r["side_fid"] == args.focus_side_fid
    ]
    focus = sorted(focus, key=lambda r: (r["match_any_rate"], math.isfinite(r["lift"]), r["lift"]), reverse=True)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    cols = [
        "item_fid", "domain", "side_fid", "rows", "match_count_abs",
        "recent_match_count_abs_2h", "match_any_rate", "match_count_mean",
        "recent_match_any_2h_rate", "recent_match_count_2h_mean",
        "matched_last_gap_mean", "matched_last_gap_p50", "matched_last_gap_p90",
        "positive_rate_when_match", "positive_rate_when_no_match", "lift",
    ]
    lines = [
        "# Pair / Target-History Match EDA v2",
        "",
        f"- data_path: `{args.data_path}`",
        f"- schema_path: `{schema_path}`",
        f"- parquet_files_scanned: `{len(parquet_files)}`",
        f"- rows_scanned: `{len(rows_raw)}`",
        f"- min_match_count: `{args.min_match_count}`",
        f"- min_recent_match_count: `{args.min_recent_match_count}`",
        f"- focus: `{args.focus_domain}` side_fid=`{args.focus_side_fid}`",
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
        "## Stable Candidate Pairs",
        "",
        "Filter: `match_count_abs >= min_match_count`, "
        "`recent_match_count_abs_2h >= min_recent_match_count`, "
        "`match_any_rate >= 0.005`, `lift > 1.05`, and positive rate improves.",
        "",
        _markdown_table(stable[:args.top_k], cols),
        "",
        "## Unstable High-Lift Low-Count Pairs",
        "",
        _markdown_table(unstable[:args.top_k], cols),
        "",
        f"## Focus Table: {args.focus_domain} side_fid={args.focus_side_fid}",
        "",
        _markdown_table(focus[:args.top_k], cols),
        "",
        "## Top Pairs By Lift",
        "",
        _markdown_table(rows_by_lift[:args.top_k], cols),
        "",
        "## Top Pairs By Match Rate",
        "",
        _markdown_table(rows_by_match[:args.top_k], cols),
        "",
        "## Top Pairs By Recent 2h Match Rate",
        "",
        _markdown_table(rows_by_recent[:args.top_k], cols),
    ])

    if args.split_by_valid_tail:
        split_idx = int(len(rows_raw) * 0.9)
        first_stats = _accumulate_rows(rows_raw[:split_idx], item_fids, seq_info)
        tail_stats = _accumulate_rows(rows_raw[split_idx:], item_fids, seq_info)
        first_rows = {k: v.as_row(*k) for k, v in first_stats.items()}
        tail_rows = {k: v.as_row(*k) for k, v in tail_stats.items()}
        stability_rows: List[Dict[str, Any]] = []
        for r in stable[:args.top_k]:
            key = (r["item_fid"], r["domain"], r["side_fid"])
            first = first_rows[key]
            tail = tail_rows[key]
            stability_rows.append({
                "item_fid": r["item_fid"],
                "domain": r["domain"],
                "side_fid": r["side_fid"],
                "first90_lift": first["lift"],
                "tail10_lift": tail["lift"],
                "first90_match_any_rate": first["match_any_rate"],
                "tail10_match_any_rate": tail["match_any_rate"],
                "first90_match_count_abs": first["match_count_abs"],
                "tail10_match_count_abs": tail["match_count_abs"],
            })
        stability_cols = [
            "item_fid", "domain", "side_fid", "first90_lift", "tail10_lift",
            "first90_match_any_rate", "tail10_match_any_rate",
            "first90_match_count_abs", "tail10_match_count_abs",
        ]
        lines.extend([
            "",
            "## First 90% vs Tail 10% Stability",
            "",
            _markdown_table(stability_rows, stability_cols),
        ])

    lines.extend([
        "",
        "## Recommendations",
        "",
        "- Prefer stable pairs with enough absolute matches and non-trivial recent 2h coverage.",
        "- Treat very high lift with tiny counts as EDA evidence only, not immediate training features.",
        "- Current P2 hypothesis should focus on target-history match plus recency for stable pairs, not global recency.",
        "- This script does not train or modify model inputs.",
    ])

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {args.output}")


def parse_args() -> argparse.Namespace:
    default_data_path = os.environ.get("TRAIN_DATA_PATH", "/data_ams/academic_training_data")
    parser = argparse.ArgumentParser(description="Target-history exact-match EDA v2")
    parser.add_argument("--data_path", type=str, default=default_data_path)
    parser.add_argument("--schema_path", type=str, default=None)
    parser.add_argument("--max_files", type=int, default=10)
    parser.add_argument("--max_rows", type=int, default=50000)
    parser.add_argument("--output", type=str, default="research/reports/pair_match_eda.md")
    parser.add_argument("--include_label", type=int, default=1, choices=[0, 1])
    parser.add_argument("--min_match_count", type=int, default=50)
    parser.add_argument("--min_recent_match_count", type=int, default=10)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--focus_side_fid", type=int, default=25)
    parser.add_argument("--focus_domain", type=str, default="seq_d")
    parser.add_argument("--split_by_valid_tail", type=int, default=0, choices=[0, 1])
    return parser.parse_args()


if __name__ == "__main__":
    run_eda(parse_args())
