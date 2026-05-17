#!/usr/bin/env python3
"""Test-aware audit for target-history match token flags.

This is read-only EDA. It checks whether P4 target-match flags are stable in
official-like narrow time tails before spending a full training run.
"""

import argparse
import csv
import glob
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pyarrow.parquet as pq


DEFAULT_SPECS: List[Dict[str, Any]] = [
    {"item_fid": 12, "domain": "seq_d", "side_fid": 25},
    {"item_fid": 13, "domain": "seq_d", "side_fid": 25},
    {"item_fid": 9, "domain": "seq_d", "side_fid": 25},
    {"item_fid": 83, "domain": "seq_d", "side_fid": 25},
    {"item_fid": 6, "domain": "seq_d", "side_fid": 24},
]


@dataclass
class SampleSpecResult:
    flag: bool
    matched_count: int
    matched_rate: float
    latest_gap: Optional[float]


@dataclass
class RowResult:
    row_idx: int
    timestamp: int
    label: Optional[int]
    spec_results: Dict[str, SampleSpecResult]


@dataclass
class SpecAgg:
    rows: int = 0
    true_count: int = 0
    pos_true: int = 0
    total_true: int = 0
    pos_false: int = 0
    total_false: int = 0
    matched_counts: List[int] = field(default_factory=list)
    matched_rates: List[float] = field(default_factory=list)
    latest_gaps: List[float] = field(default_factory=list)

    def update(self, result: SampleSpecResult, label: Optional[int]) -> None:
        self.rows += 1
        self.matched_counts.append(int(result.matched_count))
        self.matched_rates.append(float(result.matched_rate))
        if result.latest_gap is not None and math.isfinite(result.latest_gap):
            self.latest_gaps.append(float(result.latest_gap))
        if result.flag:
            self.true_count += 1
        if label is None:
            return
        if result.flag:
            self.total_true += 1
            self.pos_true += int(label)
        else:
            self.total_false += 1
            self.pos_false += int(label)

    def as_row(self, split: str, spec_name: str) -> Dict[str, Any]:
        true_rate = self.true_count / max(self.rows, 1)
        pos_true_rate = self.pos_true / self.total_true if self.total_true else math.nan
        pos_false_rate = self.pos_false / self.total_false if self.total_false else math.nan
        lift = (
            pos_true_rate / pos_false_rate
            if pos_false_rate and math.isfinite(pos_true_rate)
            else math.nan
        )
        return {
            "split": split,
            "spec": spec_name,
            "rows": self.rows,
            "true_count": self.true_count,
            "true_rate": true_rate,
            "positive_rate_when_true": pos_true_rate,
            "positive_rate_when_false": pos_false_rate,
            "lift": lift,
            "avg_matched_token_count": _mean(self.matched_counts),
            "p50_matched_token_count": _percentile(self.matched_counts, 50),
            "p90_matched_token_count": _percentile(self.matched_counts, 90),
            "p99_matched_token_count": _percentile(self.matched_counts, 99),
            "avg_matched_token_rate": _mean(self.matched_rates),
            "latest_gap_p50": _percentile(self.latest_gaps, 50),
            "latest_gap_p90": _percentile(self.latest_gaps, 90),
            "latest_gap_p99": _percentile(self.latest_gaps, 99),
        }


def _fmt(value: Any, digits: int = 6) -> str:
    if isinstance(value, float):
        if not math.isfinite(value):
            return "nan"
        return f"{value:.{digits}f}"
    return str(value)


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else math.nan


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return math.nan
    vals = sorted(float(v) for v in values)
    if len(vals) == 1:
        return vals[0]
    rank = (len(vals) - 1) * pct / 100.0
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return vals[lo]
    frac = rank - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def _markdown_table(rows: Sequence[Dict[str, Any]], cols: Sequence[str]) -> str:
    if not rows:
        return "None."
    out = ["| " + " | ".join(cols) + " |"]
    out.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for row in rows:
        out.append("| " + " | ".join(_fmt(row.get(c, "")) for c in cols) + " |")
    return "\n".join(out)


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
    return [v for v in _int_values(value) if v > 0]


def _safe_label(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(int(value) == 2)
    except (TypeError, ValueError):
        return None


def _parse_specs(raw: str) -> List[Dict[str, Any]]:
    if not raw:
        return list(DEFAULT_SPECS)
    values = json.loads(raw)
    out: List[Dict[str, Any]] = []
    for spec in values:
        out.append({
            "item_fid": int(spec["item_fid"]),
            "domain": str(spec["domain"]),
            "side_fid": int(spec["side_fid"]),
        })
    return out


def _spec_name(spec: Dict[str, Any]) -> str:
    return f"{spec['item_fid']}|{spec['domain']}|{spec['side_fid']}"


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


def _load_schema(data_path: str, schema_path: Optional[str]) -> Dict[str, Any]:
    path = schema_path or os.path.join(data_path, "schema.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _scan_rows(args: argparse.Namespace, schema: Dict[str, Any], specs: List[Dict[str, Any]]) -> Tuple[List[RowResult], int]:
    seq_info = _schema_seq_info(schema)
    needed_cols = ["timestamp", "label_type"]
    for spec in specs:
        needed_cols.append(f"item_int_feats_{spec['item_fid']}")
        info = seq_info.get(spec["domain"])
        if info:
            needed_cols.append(f"{info['prefix']}_{spec['side_fid']}")
            if info["ts_fid"] is not None:
                needed_cols.append(f"{info['prefix']}_{info['ts_fid']}")

    files = _iter_parquet_files(args.data_path, args.max_files, args.recursive_files)
    print(f"[test_aware_feature_audit] found parquet files={len(files)}", flush=True)

    rows: List[RowResult] = []
    last_progress = 0
    for file_path in files:
        if len(rows) >= args.max_rows:
            break
        pf = pq.ParquetFile(file_path)
        available = set(pf.schema_arrow.names)
        columns = [c for c in sorted(set(needed_cols)) if c in available]
        for batch in pf.iter_batches(batch_size=2048, columns=columns):
            remaining = args.max_rows - len(rows)
            if remaining <= 0:
                break
            take_n = min(batch.num_rows, remaining)
            col_values = {
                name: batch.column(batch.schema.get_field_index(name)).to_pylist()[:take_n]
                for name in columns
            }
            for i in range(take_n):
                root_ts_values = col_values.get("timestamp", [0] * take_n)
                try:
                    root_ts = int(root_ts_values[i] or 0)
                except (TypeError, ValueError):
                    root_ts = 0
                label = _safe_label(col_values.get("label_type", [None] * take_n)[i])
                spec_results: Dict[str, SampleSpecResult] = {}
                for spec in specs:
                    name = _spec_name(spec)
                    item_col = f"item_int_feats_{spec['item_fid']}"
                    info = seq_info.get(spec["domain"])
                    if info is None:
                        spec_results[name] = SampleSpecResult(False, 0, 0.0, None)
                        continue
                    side_col = f"{info['prefix']}_{spec['side_fid']}"
                    ts_col = f"{info['prefix']}_{info['ts_fid']}" if info["ts_fid"] is not None else None
                    targets = set(_positive_values(col_values.get(item_col, [None] * take_n)[i]))
                    side_vals = _int_values(col_values.get(side_col, [None] * take_n)[i])
                    seq_ts = _int_values(col_values.get(ts_col, [None] * take_n)[i]) if ts_col else []
                    spec_results[name] = _calc_spec_result(root_ts, targets, side_vals, seq_ts)
                rows.append(RowResult(len(rows), root_ts, label, spec_results))
                if len(rows) - last_progress >= 10000:
                    last_progress = len(rows)
                    print(f"[test_aware_feature_audit] rows_scanned={len(rows)}", flush=True)
            if len(rows) >= args.max_rows:
                break
    print(f"[test_aware_feature_audit] accumulate end rows_scanned={len(rows)}", flush=True)
    return rows, len(files)


def _calc_spec_result(
    root_ts: int,
    targets: set,
    side_vals: Sequence[int],
    seq_ts: Sequence[int],
) -> SampleSpecResult:
    if not targets or not side_vals:
        return SampleSpecResult(False, 0, 0.0, None)
    matched_gaps: List[int] = []
    historical_len = 0
    use_ts = bool(seq_ts) and root_ts > 0
    max_len = min(len(side_vals), len(seq_ts)) if use_ts else len(side_vals)
    for j in range(max_len):
        side = int(side_vals[j])
        if use_ts:
            ts = int(seq_ts[j])
            if ts <= 0 or ts > root_ts:
                continue
            gap = root_ts - ts
            historical_len += 1
            if side > 0 and side in targets:
                matched_gaps.append(gap)
        else:
            historical_len += 1
            if side > 0 and side in targets:
                matched_gaps.append(-1)
    matched_count = len(matched_gaps)
    latest_gap = min(matched_gaps) if use_ts and matched_gaps else None
    return SampleSpecResult(
        flag=matched_count > 0,
        matched_count=matched_count,
        matched_rate=matched_count / max(historical_len, 1),
        latest_gap=latest_gap,
    )


def _split_rows(rows: List[RowResult]) -> Dict[str, List[RowResult]]:
    n = len(rows)
    ordered = list(rows)
    max_ts = max((r.timestamp for r in rows), default=0)
    return {
        "full": ordered,
        "first90": ordered[: int(n * 0.90)],
        "tail10": ordered[int(n * 0.90):],
        "tail5": ordered[int(n * 0.95):],
        "tail1": ordered[int(n * 0.99):],
        "tail_2h": [r for r in rows if max_ts - r.timestamp <= 7200],
        "tail_6h": [r for r in rows if max_ts - r.timestamp <= 21600],
        "tail_24h": [r for r in rows if max_ts - r.timestamp <= 86400],
    }


def _summarize_split(split: str, rows: List[RowResult], spec_names: Sequence[str]) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    print(f"[test_aware_feature_audit] split start {split} rows={len(rows)}", flush=True)
    labels = [r.label for r in rows if r.label is not None]
    label_mean = sum(labels) / len(labels) if labels else math.nan
    split_row = {"split": split, "rows": len(rows), "label_mean": label_mean}

    spec_rows: List[Dict[str, Any]] = []
    for name in spec_names:
        agg = SpecAgg()
        for row in rows:
            agg.update(row.spec_results[name], row.label)
        spec_rows.append(agg.as_row(split, name))

    bucket_aggs: Dict[str, Dict[str, int]] = {
        "0": {"rows": 0, "pos": 0, "label_rows": 0},
        "1": {"rows": 0, "pos": 0, "label_rows": 0},
        "2": {"rows": 0, "pos": 0, "label_rows": 0},
        "3+": {"rows": 0, "pos": 0, "label_rows": 0},
    }
    any_agg = SpecAgg()
    for row in rows:
        num_on = sum(1 for name in spec_names if row.spec_results[name].flag)
        bucket = "3+" if num_on >= 3 else str(num_on)
        bucket_aggs[bucket]["rows"] += 1
        if row.label is not None:
            bucket_aggs[bucket]["label_rows"] += 1
            bucket_aggs[bucket]["pos"] += int(row.label)
        any_agg.update(SampleSpecResult(num_on > 0, num_on, 0.0, None), row.label)

    any_row = any_agg.as_row(split, "any_of_all_specs")
    spec_rows.append(any_row)

    bucket_rows = []
    for bucket, stats in bucket_aggs.items():
        bucket_rows.append({
            "split": split,
            "num_flags_on": bucket,
            "rows": stats["rows"],
            "row_rate": stats["rows"] / max(len(rows), 1),
            "positive_rate": (
                stats["pos"] / stats["label_rows"]
                if stats["label_rows"] else math.nan
            ),
        })
    print(f"[test_aware_feature_audit] split end {split}", flush=True)
    return split_row, spec_rows, bucket_rows


def _classify_specs(spec_rows: Sequence[Dict[str, Any]], base_specs: Sequence[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_key = {(r["split"], r["spec"]): r for r in spec_rows}
    stability_rows: List[Dict[str, Any]] = []
    strong: List[Dict[str, Any]] = []
    risky: List[Dict[str, Any]] = []
    for spec in base_specs:
        full = by_key.get(("full", spec), {})
        tail2 = by_key.get(("tail_2h", spec), {})
        tail6 = by_key.get(("tail_6h", spec), {})
        tail10 = by_key.get(("tail10", spec), {})
        row = {
            "spec": spec,
            "full_true_rate": full.get("true_rate", math.nan),
            "full_lift": full.get("lift", math.nan),
            "tail_2h_true_rate": tail2.get("true_rate", math.nan),
            "tail_2h_lift": tail2.get("lift", math.nan),
            "tail_6h_true_rate": tail6.get("true_rate", math.nan),
            "tail_6h_lift": tail6.get("lift", math.nan),
            "tail10_true_rate": tail10.get("true_rate", math.nan),
            "tail10_lift": tail10.get("lift", math.nan),
        }
        stable = (
            _gt(row["tail_2h_lift"], 1.05)
            and _gt(row["tail_6h_lift"], 1.05)
            and _gt(row["tail10_lift"], 1.05)
            and _between(row["tail_2h_true_rate"], 0.005, 0.95)
            and _between(row["tail_6h_true_rate"], 0.005, 0.95)
            and _between(row["tail10_true_rate"], 0.005, 0.95)
        )
        is_risky = (
            _gt(row["full_lift"], 1.05)
            and (_le(row["tail_2h_lift"], 1.0) or _le(row["tail_6h_lift"], 1.0))
        ) or (
            _le(row["tail_2h_true_rate"], 0.001)
            or _gt(row["tail_2h_true_rate"], 0.98)
        )
        row["stable"] = int(stable)
        row["risky"] = int(is_risky)
        stability_rows.append(row)
        if stable:
            strong.append(row)
        if is_risky:
            risky.append(row)
    return stability_rows, strong, risky


def _gt(value: Any, threshold: float) -> bool:
    return isinstance(value, float) and math.isfinite(value) and value > threshold


def _le(value: Any, threshold: float) -> bool:
    return isinstance(value, float) and math.isfinite(value) and value <= threshold


def _between(value: Any, lo: float, hi: float) -> bool:
    return isinstance(value, float) and math.isfinite(value) and lo <= value <= hi


def _write_report(
    args: argparse.Namespace,
    parquet_files_scanned: int,
    rows: List[RowResult],
    split_rows: List[Dict[str, Any]],
    spec_rows: List[Dict[str, Any]],
    bucket_rows: List[Dict[str, Any]],
    stability_rows: List[Dict[str, Any]],
    strong: List[Dict[str, Any]],
    risky: List[Dict[str, Any]],
) -> None:
    output = args.output
    out_dir = os.path.dirname(output) or "."
    os.makedirs(out_dir, exist_ok=True)
    print(f"[test_aware_feature_audit] write report start {output}", flush=True)

    split_cols = ["split", "rows", "label_mean"]
    spec_cols = [
        "split", "spec", "rows", "true_count", "true_rate",
        "positive_rate_when_true", "positive_rate_when_false", "lift",
        "avg_matched_token_count", "p50_matched_token_count",
        "p90_matched_token_count", "p99_matched_token_count",
        "avg_matched_token_rate", "latest_gap_p50", "latest_gap_p90",
        "latest_gap_p99",
    ]
    bucket_cols = ["split", "num_flags_on", "rows", "row_rate", "positive_rate"]
    stability_cols = [
        "spec", "full_true_rate", "full_lift", "tail_2h_true_rate",
        "tail_2h_lift", "tail_6h_true_rate", "tail_6h_lift",
        "tail10_true_rate", "tail10_lift", "stable", "risky",
    ]

    if args.write_csv:
        _write_csv(os.path.join(out_dir, "test_aware_feature_audit_specs.csv"), spec_rows, spec_cols)
        _write_csv(os.path.join(out_dir, "test_aware_feature_audit_splits.csv"), split_rows, split_cols)
        _write_csv(os.path.join(out_dir, "test_aware_feature_audit_num_flags.csv"), bucket_rows, bucket_cols)

    top_specs = sorted(
        [r for r in spec_rows if r["split"] in {"full", "tail_2h", "tail_6h", "tail10"}],
        key=lambda r: (float("-inf") if not isinstance(r.get("lift"), float) or not math.isfinite(r.get("lift")) else r["lift"]),
        reverse=True,
    )[: args.summary_top_k]

    lines = [
        "# Test-Aware Feature Audit",
        "",
        "## Compact Summary",
        f"- rows_scanned: {len(rows)}",
        f"- parquet_files_scanned: {parquet_files_scanned}",
        f"- strong_and_stable_specs: {len(strong)}",
        f"- risky_specs: {len(risky)}",
        "- log-friendly preview:",
        "  - awk '/^## Compact Summary/{flag=1} /^## Spec Details/{flag=0} flag{print}' test_aware_feature_audit.md",
        "  - head -80 test_aware_feature_audit_specs.csv",
        "",
        "## Recommended P4 Specs",
        _markdown_table(strong, stability_cols),
        "",
        "## Split Stability",
        _markdown_table(stability_rows, stability_cols),
        "",
        "## Risky Specs",
        _markdown_table(risky, stability_cols),
        "",
        "## Spec Details",
        _markdown_table(top_specs, spec_cols),
        "",
        "## Num Flags Bucket",
        _markdown_table(bucket_rows, bucket_cols),
        "",
        "## Split Summary",
        _markdown_table(split_rows, split_cols),
        "",
    ]
    with open(output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[test_aware_feature_audit] write report end {output}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=os.environ.get("TRAIN_DATA_PATH", "/data_ams/academic_training_data"))
    parser.add_argument("--schema_path", default="")
    parser.add_argument("--max_files", type=int, default=1000)
    parser.add_argument("--max_rows", type=int, default=200000)
    parser.add_argument("--recursive_files", type=int, default=1, choices=[0, 1])
    parser.add_argument("--specs_json", default="")
    parser.add_argument("--write_csv", type=int, default=1, choices=[0, 1])
    parser.add_argument("--summary_top_k", type=int, default=50)
    parser.add_argument("--output", default=os.path.join(os.environ.get("TRAIN_CKPT_PATH", "research/reports"), "test_aware_feature_audit.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = _parse_specs(args.specs_json)
    schema = _load_schema(args.data_path, args.schema_path or None)
    print("[test_aware_feature_audit] accumulate start", flush=True)
    rows, parquet_files_scanned = _scan_rows(args, schema, specs)
    spec_names = [_spec_name(s) for s in specs]

    split_rows: List[Dict[str, Any]] = []
    spec_rows: List[Dict[str, Any]] = []
    bucket_rows: List[Dict[str, Any]] = []
    for split, split_data in _split_rows(rows).items():
        split_row, split_spec_rows, split_bucket_rows = _summarize_split(split, split_data, spec_names)
        split_rows.append(split_row)
        spec_rows.extend(split_spec_rows)
        bucket_rows.extend(split_bucket_rows)

    stability_rows, strong, risky = _classify_specs(spec_rows, spec_names)
    _write_report(
        args=args,
        parquet_files_scanned=parquet_files_scanned,
        rows=rows,
        split_rows=split_rows,
        spec_rows=spec_rows,
        bucket_rows=bucket_rows,
        stability_rows=stability_rows,
        strong=strong,
        risky=risky,
    )


if __name__ == "__main__":
    main()
