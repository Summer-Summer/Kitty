"""
Recompute metrics from existing log files and rebuild results.csv.

Parses every log in the logs/ directory, extracts per-run e2e times,
then computes the full set of metrics and writes results.csv.
"""

import csv
import os
import re
import math
from pathlib import Path

SCRIPT_DIR  = Path(__file__).parent
LOGS_DIR    = SCRIPT_DIR / "logs"
CSV_OUT     = SCRIPT_DIR / "results.csv"

# Internal log name → display name
METHOD_TO_MODE = {"0": "Kitty-Pro", "1": "HF_Static_FP16", "2": "HF_Dynamic_FP16", "3": "HF_KIVI_INT4"}

# Full display order (including baselines not produced by this benchmark)
MODE_ORDER = [
    "HF_Static_FP16",
    "HF_Dynamic_FP16",
    "HF_KIVI_INT4",
    "Kitty-Pro",
    "SGLang BF16",
    "BDQ (fused) - Finetuned - V2",
]

CSV_HEADER = [
    "mode", "batch_size", "input_tokens", "output_tokens",
    "ttft_s", "e2e_s",
    "job_level_tps",
    "otps",
    "actual_qps",
    "per_gpu_tps_mean",
    "per_gpu_tps_stdev",
]

# Regex patterns
RE_LOGFILE   = re.compile(r"job_\d+_gpu\d+_method(\d+)_bs(\d+)\.log")
RE_RUN_LINE  = re.compile(r"->\s*e2e=([\d.]+)s\s+ttft=([\d.]+)s")
RE_PREFILL   = re.compile(r"Prefill tokens:\s*(\d+)")
RE_OUTPUT    = re.compile(r"Output tokens:\s*(\d+)")
RE_AVG_E2E   = re.compile(r"Model\.generate\(\) average execution time:\s*([\d.]+) seconds\.")
RE_AVG_TTFT  = re.compile(r"Average TTFT \(prefill\):\s*([\d.]+) seconds\.")


def parse_log(log_path: Path) -> dict:
    """Returns a dict always; sets failed=True when metrics are unavailable."""
    text = log_path.read_text(errors="replace")

    m_prefill  = RE_PREFILL.search(text)
    m_output   = RE_OUTPUT.search(text)
    m_avg_e2e  = RE_AVG_E2E.search(text)
    m_avg_ttft = RE_AVG_TTFT.search(text)

    input_tokens = int(m_prefill.group(1)) if m_prefill else None

    if not (m_output and m_avg_e2e and m_avg_ttft):
        print(f"  INCOMPLETE (OOM or crashed): {log_path.name}")
        return {"failed": True, "input_tokens": input_tokens}

    run_e2e  = [float(m.group(1)) for m in RE_RUN_LINE.finditer(text)]
    run_ttft = [float(m.group(2)) for m in RE_RUN_LINE.finditer(text)]

    output_tokens = int(m_output.group(1))

    if run_e2e:
        avg_e2e  = sum(run_e2e)  / len(run_e2e)
        avg_ttft = sum(run_ttft) / len(run_ttft)
    else:
        # Old logs without per-run lines — fall back to summary line
        avg_e2e  = float(m_avg_e2e.group(1))
        avg_ttft = float(m_avg_ttft.group(1))
        run_e2e  = [avg_e2e]

    return {
        "failed":        False,
        "input_tokens":  input_tokens,
        "output_tokens": output_tokens,
        "avg_e2e":       avg_e2e,
        "avg_ttft":      avg_ttft,
        "run_e2e":       run_e2e,
    }


def compute_metrics(data: dict, batch_size: int) -> dict:
    avg_e2e       = data["avg_e2e"]
    avg_ttft      = data["avg_ttft"]
    input_tokens  = data["input_tokens"]
    output_tokens = data["output_tokens"]
    run_e2e       = data["run_e2e"]

    job_level_tps = batch_size * output_tokens / avg_e2e
    otps          = output_tokens / (avg_e2e - avg_ttft) if (avg_e2e - avg_ttft) > 0 else math.inf
    actual_qps    = batch_size / avg_e2e

    per_run_tps   = [batch_size * output_tokens / t for t in run_e2e]
    tps_mean      = sum(per_run_tps) / len(per_run_tps)
    variance      = sum((x - tps_mean) ** 2 for x in per_run_tps) / len(per_run_tps)
    tps_stdev     = math.sqrt(variance)

    return {
        "input_tokens":       input_tokens,
        "output_tokens":      output_tokens,
        "ttft_s":             avg_ttft,
        "e2e_s":              avg_e2e,
        "job_level_tps":      job_level_tps,
        "otps":               otps,
        "actual_qps":         actual_qps,
        "per_gpu_tps_mean":   tps_mean,
        "per_gpu_tps_stdev":  tps_stdev,
    }


def main():
    log_files = sorted(LOGS_DIR.glob("*.log"))
    if not log_files:
        print(f"No log files found in {LOGS_DIR}")
        return

    rows = []
    for log_path in log_files:
        m = RE_LOGFILE.match(log_path.name)
        if not m:
            print(f"  SKIP (unrecognized filename): {log_path.name}")
            continue

        method_id  = m.group(1)
        batch_size = int(m.group(2))
        mode       = METHOD_TO_MODE.get(method_id, f"method{method_id}")

        print(f"Parsing {log_path.name} ...")
        data = parse_log(log_path)

        if data.get("failed"):
            # Keep row with -- for all metric columns
            rows.append({
                "mode": mode, "batch_size": batch_size,
                "input_tokens":      data["input_tokens"] if data["input_tokens"] else "--",
                "output_tokens":     "--",
                "ttft_s":            "--",
                "e2e_s":             "--",
                "job_level_tps":     "--",
                "otps":              "--",
                "actual_qps":        "--",
                "per_gpu_tps_mean":  "--",
                "per_gpu_tps_stdev": "--",
                "_failed":           True,
            })
            continue

        metrics = compute_metrics(data, batch_size)
        rows.append({"mode": mode, "batch_size": batch_size, **metrics})

    # Build lookup: (mode, batch_size) -> row
    data_map = {(r["mode"], r["batch_size"]): r for r in rows}
    batch_sizes = sorted({r["batch_size"] for r in rows})
    mode_rank = {m: i for i, m in enumerate(MODE_ORDER)}

    # Expand: mode-first order, then batch_size within each mode
    BLANK_ROW = {
        "input_tokens": "--", "output_tokens": "--",
        "ttft_s": "--", "e2e_s": "--",
        "job_level_tps": "--", "otps": "--", "actual_qps": "--",
        "per_gpu_tps_mean": "--", "per_gpu_tps_stdev": "--",
    }

    final_rows = []
    for mode in MODE_ORDER:
        for bs in batch_sizes:
            r = data_map.get((mode, bs), {**BLANK_ROW, "_blank": True})
            final_rows.append({"mode": mode, "batch_size": bs, **r})

    def fmt(r):
        def f(key, fmt_str):
            v = r.get(key, "--")
            return f"{v:{fmt_str}}" if isinstance(v, float) else str(v)
        return [
            r["mode"], r["batch_size"],
            r.get("input_tokens", "--"),
            r.get("output_tokens", "--"),
            f(  "ttft_s",          ".6f"),
            f(  "e2e_s",           ".6f"),
            f(  "job_level_tps",   ".2f"),
            f(  "otps",            ".2f"),
            f(  "actual_qps",      ".4f"),
            f(  "per_gpu_tps_mean",".2f"),
            f(  "per_gpu_tps_stdev",".2f"),
        ]

    with open(CSV_OUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        for r in final_rows:
            writer.writerow(fmt(r))

    print(f"\nWrote {len(final_rows)} rows to {CSV_OUT}")


if __name__ == "__main__":
    main()
