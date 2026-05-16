#!/usr/bin/env python3
"""EDA for time/recency signals before training new time features.

This script is read-only. It checks whether relative recency is useful globally
or only when conditioned on target-history exact matches. The goal is to avoid
another blind global time/recency experiment and identify stable candidates for
a future P3_target_matched_recency feature.
"""

import argparse
import csv
import glob
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow.parquet as pq


DEFAULT_FOCUS_PAIRS: List[List[Any]] = [
    [12, "seq_d", 25],
    [13, "seq_d", 25],
    [9, "seq_d", 25],
    [5, "seq_d", 25],
    [83, "seq_d", 25],
    [81, "seq_d", 25],
    [6, "seq_d", 24],
]

GAP_WINDOWS: List[Tuple[str, int]] = [
    ("30m", 1800),
    ("2h", 7200),
    ("6h", 21600),
    ("1d", 86400),
    ("3d", 259200),
    ("7d", 604800),
    ("30d", 2592000),
]


@dataclass
class BinaryLiftStats:
    rows: int = 0
    true_rows: int = 0
    pos_true: int = 0
    total_true: int = 0
    pos_false: int = 0
    total_false: int = 0

    def update(self, flag: bool, label: Optional[int]) -> None:
        self.rows += 1
        if flag:
            self.true_rows += 1
        if label is None:
            return
        if flag:
            self.total_true += 1
            self.pos_true += int(label)
        else:
            self.total_false += 1
            self.pos_false += int(label)

    def as_row(self, name: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        pos_true_rate = self.pos_true / self.total_true if self.total_true else math.nan
        pos_false_rate = self.pos_false / self.total_false if self.total_false else math.nan
        lift = (
            pos_true_rate / pos_false_rate
            if pos_false_rate and math.isfinite(pos_true_rate)
            else math.nan
        )
        row = {
            "name": name,
            "rows": self.rows,
            "true_count": self.true_rows,
            "true_rate": self.true_rows / max(self.rows, 1),
            "positive_rate_when_true": pos_true_rate,
            "positive_rate_when_false": pos_false_rate,
            "lift": lift,
        }
        if extra:
            row.update(extra)
        return row


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


def _write_csv(path: str, rows: Sequence[Dict[str, Any]], cols: Sequence[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(cols), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in cols})


def _int_values(value: Any) -> List[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        out: List[int] = []
        for x in value:
            try:
                out.append(int(x) if x is not None else 0)
            except (TypeError, ValueError):
                out.append(0)
        return out
    try:
        return [int(value)]
    except (TypeError, ValueError):
        return []


def _positive_values(value: Any) -> List[int]:
    return [x for x in _int_values(value) if x > 0]


def _safe_label(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(int(value) == 2)
    except (TypeError, ValueError):
        return None


def _parse_pairs(raw: str) -> List[List[Any]]:
    if not raw:
        return DEFAULT_FOCUS_PAIRS
    try:
        pairs = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid --focus_pairs_json: {exc}") from exc
    out: List[List[Any]] = []
    for p in pairs:
        if not isinstance(p, list) or len(p) != 3:
            raise ValueError(f"Pair must be [item_fid, domain, side_fid], got {p!r}")
        out.append([int(p[0]), str(p[1]), int(p[2])])
    return out


def _iter_parquet_files(data_path: str, max_files: int, recursive_files: int) -> List[str]:
    direct = sorted(glob.glob(os.path.join(data_path, "*.parquet")))
    files = list(direct)
    if recursive_files and len(files) < max_files:
        seen = set(files)
        recursive = sorted(glob.glob(os.path.join(data_path, "**", "*.parquet"), recursive=True))
        files.extend([p for p in recursive if p not in seen])
    return files[:max_files]


def _schema_seq_info(schema: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    info: Dict[str, Dict[str, Any]] = {}
    for domain, cfg in sorted(schema.get("seq", {}).items()):
        info[domain] = {
            "prefix": cfg["prefix"],
            "ts_fid": cfg.get("ts_fid"),
        }
    return info


def _scan_rows(
    data_path: str,
    schema: Dict[str, Any],
    pairs: Sequence[Sequence[Any]],
    max_files: int,
    max_rows: int,
    include_label: bool,
    recursive_files: int,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    seq_info = _schema_seq_info(schema)
    needed_cols = ["timestamp"]
    if include_label:
        needed_cols.append("label_type")
    for item_fid, domain, side_fid in pairs:
        needed_cols.append(f"item_int_feats_{int(item_fid)}")
        if domain in seq_info:
            prefix = seq_info[domain]["prefix"]
            ts_fid = seq_info[domain]["ts_fid"]
            if ts_fid is not None:
                needed_cols.append(f"{prefix}_{ts_fid}")
            needed_cols.append(f"{prefix}_{int(side_fid)}")
    for domain, info in seq_info.items():
        prefix = info["prefix"]
        ts_fid = info["ts_fid"]
        if ts_fid is not None:
            needed_cols.append(f"{prefix}_{ts_fid}")

    rows: List[Dict[str, Any]] = []
    files = _iter_parquet_files(data_path, max_files, recursive_files)
    print(f"[time_signal_eda] found parquet files={len(files)}", flush=True)
    last_progress = 0
    for file_path in files:
        print(f"[time_signal_eda] reading {file_path}", flush=True)
        pf = pq.ParquetFile(file_path)
        available = set(pf.schema_arrow.names)
        columns = [c for c in sorted(set(needed_cols)) if c in available]
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
                row = {
                    "timestamp": col_values.get("timestamp", [0] * take_n)[i],
                    "label": _safe_label(col_values.get("label_type", [None] * take_n)[i])
                    if include_label else None,
                    "cols": {name: col_values[name][i] for name in columns if name not in {"timestamp", "label_type"}},
                }
                rows.append(row)
                if len(rows) - last_progress >= 5000:
                    last_progress = len(rows)
                    print(f"[time_signal_eda] rows_scanned={len(rows)}", flush=True)
            if len(rows) >= max_rows:
                break
        if len(rows) >= max_rows:
            break
    return rows, files


def _historical_gaps(root: int, seq_ts: Sequence[int]) -> Tuple[List[int], int]:
    gaps: List[int] = []
    future_or_invalid = 0
    for ts in seq_ts:
        if ts <= 0:
            future_or_invalid += 1
            continue
        if ts > root:
            future_or_invalid += 1
            continue
        gaps.append(root - ts)
    return gaps, future_or_invalid


def _matched_gaps(root: int, seq_ts: Sequence[int], side_vals: Sequence[int], targets: Sequence[int]) -> Tuple[List[int], int]:
    target_set = {v for v in targets if v > 0}
    gaps: List[int] = []
    future_or_invalid = 0
    if not target_set:
        return gaps, 0
    for pos, value in enumerate(side_vals):
        if value <= 0 or pos >= len(seq_ts):
            continue
        ts = seq_ts[pos]
        if ts <= 0 or ts > root:
            future_or_invalid += 1
            continue
        if value in target_set:
            gaps.append(root - ts)
    return gaps, future_or_invalid


def _accumulate(
    rows: Sequence[Dict[str, Any]],
    schema: Dict[str, Any],
    pairs: Sequence[Sequence[Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    seq_info = _schema_seq_info(schema)
    global_stats: Dict[Tuple[str, str], BinaryLiftStats] = {}
    pair_stats: Dict[Tuple[int, str, int, str], BinaryLiftStats] = {}
    future_invalid = 0
    total_ts_positions = 0

    for domain in seq_info:
        for window_name, _sec in GAP_WINDOWS:
            global_stats[(domain, window_name)] = BinaryLiftStats()
    for item_fid, domain, side_fid in pairs:
        for window_name, _sec in GAP_WINDOWS:
            pair_stats[(int(item_fid), str(domain), int(side_fid), window_name)] = BinaryLiftStats()

    for row in rows:
        try:
            root = int(row["timestamp"]) if row["timestamp"] is not None else 0
        except (TypeError, ValueError):
            root = 0
        label = row["label"]
        cols = row["cols"]

        for domain, info in seq_info.items():
            prefix = info["prefix"]
            ts_fid = info["ts_fid"]
            seq_ts = _int_values(cols.get(f"{prefix}_{ts_fid}")) if ts_fid is not None else []
            gaps, bad = _historical_gaps(root, seq_ts)
            future_invalid += bad
            total_ts_positions += len(seq_ts)
            for window_name, sec in GAP_WINDOWS:
                global_stats[(domain, window_name)].update(any(g <= sec for g in gaps), label)

        for item_fid, domain, side_fid in pairs:
            domain = str(domain)
            if domain not in seq_info:
                continue
            info = seq_info[domain]
            prefix = info["prefix"]
            ts_fid = info["ts_fid"]
            seq_ts = _int_values(cols.get(f"{prefix}_{ts_fid}")) if ts_fid is not None else []
            side_vals = _int_values(cols.get(f"{prefix}_{int(side_fid)}"))
            targets = _positive_values(cols.get(f"item_int_feats_{int(item_fid)}"))
            match_gaps, bad = _matched_gaps(root, seq_ts, side_vals, targets)
            future_invalid += bad
            for window_name, sec in GAP_WINDOWS:
                pair_stats[(int(item_fid), domain, int(side_fid), window_name)].update(
                    any(g <= sec for g in match_gaps),
                    label,
                )

    global_rows: List[Dict[str, Any]] = []
    for (domain, window_name), stat in global_stats.items():
        global_rows.append(stat.as_row(
            f"{domain}_recent_{window_name}",
            {"domain": domain, "window": window_name},
        ))

    pair_rows: List[Dict[str, Any]] = []
    for (item_fid, domain, side_fid, window_name), stat in pair_stats.items():
        pair_rows.append(stat.as_row(
            f"item{item_fid}_{domain}_side{side_fid}_matched_recent_{window_name}",
            {
                "item_fid": item_fid,
                "domain": domain,
                "side_fid": side_fid,
                "window": window_name,
            },
        ))

    meta = {
        "future_or_invalid_ts_filtered": future_invalid,
        "total_seq_ts_positions": total_ts_positions,
        "future_or_invalid_ts_rate": future_invalid / max(total_ts_positions, 1),
    }
    return global_rows, pair_rows, meta


def _sort_signal(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        rows,
        key=lambda r: (
            math.isfinite(r["lift"]),
            r["lift"] if math.isfinite(r["lift"]) else -math.inf,
            r["true_rate"],
        ),
        reverse=True,
    )


def run_eda(args: argparse.Namespace) -> None:
    schema_path = args.schema_path or os.path.join(args.data_path, "schema.json")
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    pairs = _parse_pairs(args.focus_pairs_json)
    rows, files = _scan_rows(
        args.data_path,
        schema,
        pairs,
        args.max_files,
        args.max_rows,
        bool(args.include_label),
        args.recursive_files,
    )
    print(f"[time_signal_eda] accumulate start rows={len(rows)}", flush=True)
    global_rows, pair_rows, meta = _accumulate(rows, schema, pairs)
    print("[time_signal_eda] accumulate end", flush=True)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    global_cols = [
        "domain", "window", "rows", "true_count", "true_rate",
        "positive_rate_when_true", "positive_rate_when_false", "lift",
    ]
    pair_cols = [
        "item_fid", "domain", "side_fid", "window", "rows", "true_count",
        "true_rate", "positive_rate_when_true", "positive_rate_when_false", "lift",
    ]

    sorted_global = _sort_signal(global_rows)
    sorted_pair = _sort_signal(pair_rows)
    stable_pair_candidates = [
        r for r in sorted_pair
        if r["true_count"] >= 10
        and r["true_rate"] >= 0.005
        and math.isfinite(r["lift"])
        and r["lift"] > 1.05
        and r["positive_rate_when_true"] > r["positive_rate_when_false"]
    ]
    recommended = [
        {
            "item_fid": r["item_fid"],
            "domain": r["domain"],
            "side_fid": r["side_fid"],
            "window": r["window"],
            "lift": r["lift"],
            "true_rate": r["true_rate"],
            "true_count": r["true_count"],
        }
        for r in stable_pair_candidates[:args.summary_top_k]
    ]
    rec_cols = ["item_fid", "domain", "side_fid", "window", "lift", "true_rate", "true_count"]

    lines = [
        "# Time / Target-Matched Recency EDA",
        "",
        "## Compact Summary",
        "",
        f"- rows_scanned: `{len(rows)}`",
        f"- parquet_files_scanned: `{len(files)}`",
        f"- future_or_invalid_ts_rate: `{meta['future_or_invalid_ts_rate']:.6f}`",
        "",
        "### Global Recency Top Signals",
        "",
        _markdown_table(sorted_global[:args.summary_top_k], global_cols),
        "",
        "### Target-Matched Recency Stable Candidates",
        "",
        _markdown_table(stable_pair_candidates[:args.summary_top_k], pair_cols),
        "",
        "### Recommended P3 Pairs / Windows",
        "",
        _markdown_table(recommended, rec_cols),
        "",
        "---",
        "",
        f"- data_path: `{args.data_path}`",
        f"- schema_path: `{schema_path}`",
        f"- parquet_files_scanned: `{len(files)}`",
        f"- rows_scanned: `{len(rows)}`",
        f"- focus_pairs: `{pairs}`",
        f"- future_or_invalid_ts_filtered: `{meta['future_or_invalid_ts_filtered']}`",
        f"- future_or_invalid_ts_rate: `{meta['future_or_invalid_ts_rate']:.6f}`",
        "",
        "## Global Domain Recency Lift",
        "",
        "This is the R01-style risk area. Strong global lift is useful only if it is stable and not driven by root-time drift.",
        "",
        _markdown_table(sorted_global, global_cols),
        "",
        "## Target-Matched Recency Lift",
        "",
        "This is the preferred next time direction: recency conditioned on target-history exact match.",
        "",
        _markdown_table(sorted_pair[:args.top_k], pair_cols),
    ]

    stability: List[Dict[str, Any]] = []
    stability_cols = [
        "item_fid", "domain", "side_fid", "window", "full_lift",
        "first90_lift", "tail10_lift", "full_true_rate", "tail10_true_rate",
    ]
    if args.split_by_valid_tail:
        split_idx = int(len(rows) * 0.9)
        global_first, pair_first, _ = _accumulate(rows[:split_idx], schema, pairs)
        global_tail, pair_tail, _ = _accumulate(rows[split_idx:], schema, pairs)
        first_map = {
            (r.get("item_fid"), r.get("domain"), r.get("side_fid"), r.get("window")): r
            for r in pair_first
        }
        tail_map = {
            (r.get("item_fid"), r.get("domain"), r.get("side_fid"), r.get("window")): r
            for r in pair_tail
        }
        for r in sorted_pair[:args.top_k]:
            key = (r.get("item_fid"), r.get("domain"), r.get("side_fid"), r.get("window"))
            first = first_map.get(key, {})
            tail = tail_map.get(key, {})
            stability.append({
                "item_fid": r.get("item_fid"),
                "domain": r.get("domain"),
                "side_fid": r.get("side_fid"),
                "window": r.get("window"),
                "full_lift": r.get("lift"),
                "first90_lift": first.get("lift", math.nan),
                "tail10_lift": tail.get("lift", math.nan),
                "full_true_rate": r.get("true_rate"),
                "tail10_true_rate": tail.get("true_rate", math.nan),
            })
        lines.extend([
            "",
            "## Target-Matched Recency Stability: First 90% vs Tail 10%",
            "",
            _markdown_table(stability, stability_cols),
        ])

    lines.extend([
        "",
        "## How To Use This Report",
        "",
        "- Prefer target-matched recency rows with non-trivial true_rate, positive lift, and stable tail10 lift.",
        "- If global recency is strong but target-matched recency is weak, do not train another global residual directly; inspect drift first.",
        "- If target-matched 2h/1d rows are stable, the next clean experiment is P3_target_matched_recency_v1.",
        "- This script is EDA only and does not change model features.",
    ])

    print("[time_signal_eda] write report start", flush=True)
    if args.write_csv:
        _write_csv(os.path.join(out_dir or ".", "time_global_recency.csv"), sorted_global, global_cols)
        _write_csv(os.path.join(out_dir or ".", "time_target_matched_recency.csv"), sorted_pair, pair_cols)
        _write_csv(os.path.join(out_dir or ".", "time_target_matched_stability.csv"), stability, stability_cols)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[time_signal_eda] write report end output={args.output}", flush=True)


def parse_args() -> argparse.Namespace:
    default_data_path = os.environ.get("TRAIN_DATA_PATH", "/data_ams/academic_training_data")
    parser = argparse.ArgumentParser(description="Time and target-matched recency EDA")
    parser.add_argument("--data_path", default=default_data_path)
    parser.add_argument("--schema_path", default=None)
    parser.add_argument("--max_files", type=int, default=10)
    parser.add_argument("--max_rows", type=int, default=50000)
    parser.add_argument("--include_label", type=int, default=1, choices=[0, 1])
    parser.add_argument("--focus_pairs_json", default="")
    parser.add_argument("--split_by_valid_tail", type=int, default=1, choices=[0, 1])
    parser.add_argument("--top_k", type=int, default=80)
    parser.add_argument("--recursive_files", type=int, default=1, choices=[0, 1])
    parser.add_argument("--write_csv", type=int, default=1, choices=[0, 1])
    parser.add_argument("--summary_top_k", type=int, default=30)
    parser.add_argument("--output", default="research/reports/time_signal_eda.md")
    return parser.parse_args()


if __name__ == "__main__":
    run_eda(parse_args())
