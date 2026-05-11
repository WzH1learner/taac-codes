#!/usr/bin/env python3
"""
Ultimate TAIJI log parser: extracts configuration, core metrics table, full CSV, best epoch, OOM logs.
Usage: python parse_log_final.py <log_file>
"""

import re
import sys
import ast

def parse_args_line(line):
    match = re.search(r"Args:\s*(\{.*\})", line)
    if not match:
        return None
    try:
        return ast.literal_eval(match.group(1))
    except Exception as e:
        print(f"Warning: Could not parse Args: {e}", file=sys.stderr)
        return None

def parse_model_params(lines, start_idx):
    total = sparse = dense = None
    for i in range(start_idx, min(start_idx+50, len(lines))):
        line = lines[i]
        if "Total parameters:" in line:
            m = re.search(r'Total parameters:\s+([\d,]+)', line)
            if m:
                total = int(m.group(1).replace(',', ''))
        elif "Sparse params:" in line:
            m = re.search(r'Sparse params:\s+[\d]+\s+tensors,\s+([\d,]+)', line)
            if m:
                sparse = int(m.group(1).replace(',', ''))
        elif "Dense params:" in line:
            m = re.search(r'Dense params:\s+[\d]+\s+tensors,\s+([\d,]+)', line)
            if m:
                dense = int(m.group(1).replace(',', ''))
        if total is not None and sparse is not None and dense is not None:
            break
    return total, sparse, dense

def parse_metrics_full(lines):
    epochs = []
    current = {}
    epoch_avg_loss = re.compile(r'Epoch (\d+), Average Loss: ([\d.]+)')
    train_metrics = re.compile(r'TrainEpoch metrics @ step \d+:\s*(.+)')
    valid_metrics = re.compile(r'Valid metrics @ step \d+:\s*(.+)')
    reinit_line = re.compile(r'Re-initialized (\d+) high-cardinality Embeddings \(vocab>0\), kept (\d+)')
    rebuilt_line = re.compile(r'Rebuilt Adagrad optimizer after epoch \d+, old_state candidates=(\d+), reinit params=(\d+), restored optimizer state for (\d+) low-cardinality params')

    for line in lines:
        m = epoch_avg_loss.search(line)
        if m:
            if current:
                epochs.append(current)
            current = {'epoch': int(m.group(1)), 'avg_loss': float(m.group(2))}
            continue
        m = train_metrics.search(line)
        if m and current:
            parts = m.group(1).split(', ')
            for part in parts:
                if '=' in part:
                    k, v = part.split('=', 1)
                    try:
                        current[f'train_{k}'] = float(v)
                    except ValueError:
                        current[f'train_{k}'] = v
            continue
        m = valid_metrics.search(line)
        if m and current:
            parts = m.group(1).split(', ')
            for part in parts:
                if '=' in part:
                    k, v = part.split('=', 1)
                    try:
                        current[f'valid_{k}'] = float(v)
                    except ValueError:
                        current[f'valid_{k}'] = v
            continue
        m = reinit_line.search(line)
        if m and current:
            current['reinit_count'] = int(m.group(1))
            current['kept'] = int(m.group(2))
        m = rebuilt_line.search(line)
        if m and current:
            current['old_state_candidates'] = int(m.group(1))
            current['reinit_params'] = int(m.group(2))
            current['restored_count'] = int(m.group(3))
    if current:
        epochs.append(current)
    return epochs

def find_best_epoch(epochs, metric='valid_AUC', higher_is_better=True):
    best_epoch = None
    best_value = -1e9 if higher_is_better else 1e9
    for ep in epochs:
        if metric in ep:
            val = ep[metric]
            if higher_is_better:
                if val > best_value:
                    best_value = val
                    best_epoch = ep['epoch']
            else:
                if val < best_value:
                    best_value = val
                    best_epoch = ep['epoch']
    return best_epoch, best_value

def extract_oom_logs(lines):
    oom_lines = []
    for i, line in enumerate(lines):
        if 'OutOfMemoryError' in line or 'CUDA out of memory' in line:
            start = max(0, i-2)
            end = min(len(lines), i+3)
            context = lines[start:end]
            oom_lines.append(''.join(context).strip())
    return oom_lines

def extract_context_logs(lines):
    """Extract non-epoch context that explains data/model comparability."""
    context = {
        'row_group_split': [],
        'timestamp_ranges': [],
        'embedding_skips': [],
        'model_shape': [],
    }
    row_group_pat = re.compile(r'Row Group split.*')
    ts_pat = re.compile(r'Row Group (?:train|valid) timestamp range:.*')
    skip_pat = re.compile(r'emb_skip_threshold=\d+: .* skipped \d+/\d+ features')
    model_pat = re.compile(
        r'PCVRHyFormer model created: num_ns=\d+, T=\d+, d_model=\d+, '
        r'rank_mixer_mode=\w+')

    for line in lines:
        clean = line.strip()
        for key, pat in [
            ('row_group_split', row_group_pat),
            ('timestamp_ranges', ts_pat),
            ('embedding_skips', skip_pat),
            ('model_shape', model_pat),
        ]:
            m = pat.search(clean)
            if m:
                context[key].append(m.group(0))
    return context

def print_core_table(epochs):
    """Print a clean table with core metrics."""
    headers = ["Epoch", "Avg Loss", "Train grad_norm", "Train prob_mean", "Train prob_std",
               "Valid AUC", "Valid LogLoss", "Valid Brier", "Valid prob_mean", "Valid prob_std"]
    # Prepare rows
    rows = []
    for ep in epochs:
        row = [
            str(ep.get('epoch', '')),
            f"{ep.get('avg_loss', ''):.6f}" if 'avg_loss' in ep else '',
            f"{ep.get('train_grad_norm', ''):.6f}" if 'train_grad_norm' in ep else '',
            f"{ep.get('train_prob_mean', ''):.6f}" if 'train_prob_mean' in ep else '',
            f"{ep.get('train_prob_std', ''):.6f}" if 'train_prob_std' in ep else '',
            f"{ep.get('valid_AUC', ''):.6f}" if 'valid_AUC' in ep else '',
            f"{ep.get('valid_LogLoss', ''):.6f}" if 'valid_LogLoss' in ep else '',
            f"{ep.get('valid_brier', ''):.6f}" if 'valid_brier' in ep else '',
            f"{ep.get('valid_prob_mean', ''):.6f}" if 'valid_prob_mean' in ep else '',
            f"{ep.get('valid_prob_std', ''):.6f}" if 'valid_prob_std' in ep else '',
        ]
        rows.append(row)
    # Compute column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(val))
    # Print table
    sep = "+" + "+".join("-" * (w+2) for w in col_widths) + "+"
    header_row = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    print(sep)
    print(header_row)
    print(sep)
    for row in rows:
        formatted_row = "| " + " | ".join(row[i].ljust(col_widths[i]) for i in range(len(headers))) + " |"
        print(formatted_row)
    print(sep)

def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_log_final.py <log_file>", file=sys.stderr)
        sys.exit(1)

    log_file = sys.argv[1]
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: File '{log_file}' not found.", file=sys.stderr)
        sys.exit(1)

    # Find Args and model params
    args = None
    total = sparse = dense = None
    for i, line in enumerate(lines):
        if "Args:" in line and "{" in line:
            args = parse_args_line(line)
            if args:
                for j in range(i, min(i+500, len(lines))):
                    if "PCVRHyFormer model created" in lines[j]:
                        total, sparse, dense = parse_model_params(lines, j)
                        break
                break

    # Print config
    print("=" * 80)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 80)
    if args:
        for k in sorted(args.keys()):
            print(f"{k:35}: {args[k]}")
    if total:
        print(f"{'Total parameters':35}: {total:,}")
        print(f"{'Sparse parameters':35}: {sparse:,}")
        print(f"{'Dense parameters':35}: {dense:,}")
    print("=" * 80)

    context = extract_context_logs(lines)
    if any(context.values()):
        print("\n" + "=" * 80)
        print("DATA / MODEL CONTEXT")
        print("=" * 80)
        for title, key in [
            ("Row Group split", 'row_group_split'),
            ("Timestamp ranges", 'timestamp_ranges'),
            ("Embedding skip logs", 'embedding_skips'),
            ("Model token shape", 'model_shape'),
        ]:
            values = context[key]
            if values:
                print(f"\n{title}:")
                for value in values:
                    print(f"  {value}")

    epochs = parse_metrics_full(lines)
    if not epochs:
        print("No epoch metrics found in log.")
        return

    # Best epoch
    best_ep, best_auc = find_best_epoch(epochs, 'valid_AUC', higher_is_better=True)
    if best_ep is not None:
        print(f"\n>>> BEST EPOCH: {best_ep} (valid_AUC = {best_auc:.6f})\n")
    else:
        print("\n>>> No valid_AUC found to determine best epoch.\n")

    # Core table
    print("CORE METRICS TABLE")
    print_core_table(epochs)

    # Full CSV output (all metrics)
    print("\n" + "=" * 80)
    print("FULL DETAILS (CSV format, copy to spreadsheet)")
    print("=" * 80)
    all_keys = set()
    for ep in epochs:
        all_keys.update(ep.keys())
    ordered_keys = ['epoch', 'avg_loss']
    train_keys = sorted([k for k in all_keys if k.startswith('train_')])
    valid_keys = sorted([k for k in all_keys if k.startswith('valid_')])
    reinit_keys = ['reinit_count', 'kept', 'old_state_candidates', 'reinit_params', 'restored_count']
    for k in train_keys:
        ordered_keys.append(k)
    for k in valid_keys:
        ordered_keys.append(k)
    for k in reinit_keys:
        if k in all_keys:
            ordered_keys.append(k)
    print(",".join(ordered_keys))
    for ep in epochs:
        row = []
        for k in ordered_keys:
            v = ep.get(k, '')
            if isinstance(v, float):
                row.append(f"{v:.6f}")
            else:
                row.append(str(v))
        print(",".join(row))

    # Reinit summary
    print("\n" + "=" * 80)
    print("REINITIALIZATION DETAILS (per epoch)")
    print("=" * 80)
    for ep in epochs:
        if 'reinit_count' in ep:
            kept = ep.get('kept', '?')
            old = ep.get('old_state_candidates', '?')
            reinit_params = ep.get('reinit_params', '?')
            restored = ep.get('restored_count', '?')
            print(f"Epoch {ep['epoch']}: reinit {ep['reinit_count']} embeddings (kept {kept}), old_state_candidates={old}, reinit_params={reinit_params}, restored={restored}")

    # OOM logs
    oom_logs = extract_oom_logs(lines)
    if oom_logs:
        print("\n" + "=" * 80)
        print("OUT-OF-MEMORY (OOM) LOGS")
        print("=" * 80)
        for idx, log in enumerate(oom_logs, 1):
            print(f"OOM #{idx}:\n{log}\n")
    else:
        print("\nNo OOM errors found in log.")

if __name__ == "__main__":
    main()
