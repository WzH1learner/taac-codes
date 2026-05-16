#!/usr/bin/env python3
"""Print compact EDA report summaries for cloud logs.

The cloud platform makes it awkward to inspect files under TRAIN_CKPT_PATH.
This helper reads markdown/CSV outputs from pair_match_eda and time_signal_eda
and prints only the high-signal tables to stdout.
"""

import argparse
import csv
import json
import math
import os
from typing import Any, Dict, Iterable, List, Sequence


DEFAULT_FOCUS_PAIRS = [
    [12, "seq_d", 25],
    [6, "seq_d", 24],
    [13, "seq_d", 25],
    [83, "seq_d", 25],
    [9, "seq_d", 25],
    [5, "seq_d", 25],
    [81, "seq_d", 25],
]

EXPECTED_FILES = [
    "pair_match_eda_v3.md",
    "pair_match_stable.csv",
    "pair_match_focus.csv",
    "pair_match_stability.csv",
    "time_signal_eda_v3.md",
    "time_global_recency.csv",
    "time_target_matched_recency.csv",
    "time_target_matched_stability.csv",
]


def _read_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _to_float(value: Any) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return math.nan
    return x if math.isfinite(x) else math.nan


def _fmt(value: Any, digits: int = 6) -> str:
    x = _to_float(value)
    if math.isfinite(x):
        return f"{x:.{digits}f}"
    if value is None:
        return ""
    return str(value)


def _markdown_table(rows: Sequence[Dict[str, Any]], cols: Sequence[str]) -> str:
    if not rows:
        return "None."
    lines = ["| " + " | ".join(cols) + " |"]
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(c, "")) for c in cols) + " |")
    return "\n".join(lines)


def _sort_by_numeric(rows: Iterable[Dict[str, str]], key: str) -> List[Dict[str, str]]:
    return sorted(rows, key=lambda r: _to_float(r.get(key)), reverse=True)


def _parse_focus_pairs(raw: str) -> List[List[Any]]:
    if not raw:
        return DEFAULT_FOCUS_PAIRS
    pairs = json.loads(raw)
    return [[int(p[0]), str(p[1]), int(p[2])] for p in pairs]


def _matches_focus(row: Dict[str, str], focus_pairs: Sequence[Sequence[Any]]) -> bool:
    try:
        item_fid = int(float(row.get("item_fid", "")))
        domain = str(row.get("domain", ""))
        side_fid = int(float(row.get("side_fid", "")))
    except (TypeError, ValueError):
        return False
    return any(
        item_fid == int(p[0]) and domain == str(p[1]) and side_fid == int(p[2])
        for p in focus_pairs
    )


def _print_file_status(out_dir: str) -> None:
    print("## EDA Output File Check")
    print("")
    print("| file | status | size_bytes |")
    print("| --- | --- | ---: |")
    missing_pair = False
    for name in EXPECTED_FILES:
        path = os.path.join(out_dir, name)
        exists = os.path.exists(path)
        size = os.path.getsize(path) if exists else 0
        status = "exists" if exists else "missing"
        print(f"| {name} | {status} | {size} |")
        if name.startswith("pair_") and not exists:
            missing_pair = True
    print("")
    if missing_pair:
        print("[WARN] pair outputs missing; pair EDA may not have run or wrote to another dir.")
        print("")


def main() -> None:
    parser = argparse.ArgumentParser(description="Print compact EDA summaries")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--focus_pairs_json", default="")
    args = parser.parse_args()

    out_dir = args.out_dir
    focus_pairs = _parse_focus_pairs(args.focus_pairs_json)

    _print_file_status(out_dir)

    target_path = os.path.join(out_dir, "time_target_matched_recency.csv")
    stability_path = os.path.join(out_dir, "time_target_matched_stability.csv")
    global_path = os.path.join(out_dir, "time_global_recency.csv")

    target_rows = _read_csv(target_path)
    stability_rows = _read_csv(stability_path)
    global_rows = _read_csv(global_path)

    target_cols = [
        "item_fid", "domain", "side_fid", "window", "rows", "true_count",
        "true_rate", "positive_rate_when_true", "positive_rate_when_false", "lift",
    ]
    stability_cols = [
        "item_fid", "domain", "side_fid", "window", "full_lift",
        "first90_lift", "tail10_lift", "full_true_rate", "tail10_true_rate",
    ]
    global_cols = [
        "domain", "window", "rows", "true_count", "true_rate",
        "positive_rate_when_true", "positive_rate_when_false", "lift",
    ]

    print("## Target Matched Recency: Top By Lift")
    print("")
    print(_markdown_table(_sort_by_numeric(target_rows, "lift")[:args.top_k], target_cols))
    print("")

    print("## Target Matched Stability: Top By tail10_lift")
    print("")
    print(_markdown_table(_sort_by_numeric(stability_rows, "tail10_lift")[:args.top_k], stability_cols))
    print("")

    print("## Global Recency: Top By Lift")
    print("")
    print(_markdown_table(_sort_by_numeric(global_rows, "lift")[:50], global_cols))
    print("")

    focus_rows = [r for r in target_rows if _matches_focus(r, focus_pairs)]
    focus_rows = sorted(
        focus_rows,
        key=lambda r: (
            str(r.get("domain", "")),
            _to_float(r.get("item_fid")),
            _to_float(r.get("side_fid")),
            _to_float(r.get("lift")),
        ),
        reverse=True,
    )
    print("## Special Focus Pairs: All Windows")
    print("")
    print(f"focus_pairs = `{focus_pairs}`")
    print("")
    print(_markdown_table(focus_rows, target_cols))
    print("")


if __name__ == "__main__":
    main()
