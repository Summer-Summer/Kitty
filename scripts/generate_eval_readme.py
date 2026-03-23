#!/usr/bin/env python3
"""Generate README.md from accuracy_simulation/eval_results_rerun results."""

import json
import re
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "accuracy_simulation" / "eval_results_rerun"
OUTPUT_PATH = RESULTS_DIR / "README.md"

# Promote ratio columns in order
PROMOTE_RATIOS = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]
K4V2_KEY = "k4v2_promo0"  # k4v2 with promote_ratio=0 (equivalent to 100% high-precision)

# Column display labels (percentages)
COL_KEYS = PROMOTE_RATIOS + [K4V2_KEY]
COL_HEADERS = [f"{r * 100:g}%" for r in PROMOTE_RATIOS] + ["100%"]

SEL_LABELS = {
    "sel1": "sel1 (Magnitude)",
    "sel2": "sel2 (Variance)",
}

# Metric key -> display label
METRIC_LABELS = {
    "exact_match,strict-match": "Strict Match",
    "exact_match,flexible-extract": "Flexible Extract",
    "exact_match,none": "Exact Match",
    "math_verify,none": "Math Verify",
}

# Per-task metric order; falls back to auto-detection from collected data
TASK_METRICS = {
    "gsm8k_cot_llama": ["exact_match,strict-match", "exact_match,flexible-extract"],
    "minerva_math_algebra": ["exact_match,none", "math_verify,none"],
}

DIGITS = 4  # decimal places for rounding


def parse_config_dir(name: str) -> dict | None:
    """Parse directory name like kitty_g128_b128_s32_sel2_k2_v2_pb4_pr0.5"""
    m = re.fullmatch(
        r"kitty_g(\d+)_b(\d+)_s(\d+)_sel(\d+)_k(\d+)_v(\d+)_pb(\d+)_pr([\d.]+)",
        name,
    )
    if not m:
        return None
    return {
        "sel": int(m.group(4)),
        "kbits": int(m.group(5)),
        "vbits": int(m.group(6)),
        "promote_ratio": float(m.group(8)),
    }


def collect_results() -> dict:
    """
    Returns nested dict:
      results[model][sel_key][task][col_key][metric_key] = [run1, run2, ...]
    Values are the raw per-repeat scores from summary.json statistics.values.
    """
    results = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    )

    for model_dir in sorted(RESULTS_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        model = model_dir.name

        for task_dir in sorted(model_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            task = task_dir.name

            for config_dir in sorted(task_dir.iterdir()):
                if not config_dir.is_dir():
                    continue

                cfg = parse_config_dir(config_dir.name)
                if cfg is None:
                    continue

                summary_files = list(config_dir.glob("*_summary.json"))
                if not summary_files:
                    continue

                with open(summary_files[0]) as f:
                    data = json.load(f)
                stats = data.get("statistics", {})
                if not stats:
                    continue

                sel_key = f"sel{cfg['sel']}"

                if cfg["kbits"] == 4 and cfg["vbits"] == 2 and cfg["promote_ratio"] == 0.0:
                    col_key = K4V2_KEY
                else:
                    col_key = cfg["promote_ratio"]

                for metric_key in stats:
                    results[model][sel_key][task][col_key][metric_key] = (
                        stats[metric_key]["values"]
                    )

    return results


def r(val: float) -> float:
    """Round to DIGITS decimal places."""
    return round(val, DIGITS)


def fmt(val: float | None) -> str:
    return "--" if val is None else f"{val:.{DIGITS}f}"


def compute_rows(raw_values: list[float]) -> tuple[list[float], float]:
    """
    Round each run value, then compute mean of the rounded values and round again.
    Returns (rounded_runs, rounded_mean).
    """
    rounded = [r(v) for v in raw_values]
    mean = r(sum(rounded) / len(rounded))
    return rounded, mean


def generate_readme(results: dict) -> str:
    lines = [
        "# Eval Results (Rerun)",
        "",
        "Auto-generated from `accuracy_simulation/eval_results_rerun/`.",
        "",
        "**Selection methods:**",
        "- `sel1`: Magnitude-based channel selection",
        "- `sel2`: Variance-based channel selection",
        "",
        "**Columns:** promote ratio 0% → 87.5% (k2v2, promote_bit=4), "
        "then 100% (k4v2 promo0)",
        "",
        "**Values:** each run rounded to 4 decimal places; "
        "Mean computed from rounded values, then rounded again (`--` = data not available)",
        "",
    ]

    for model in sorted(results):
        lines.append(f"## {model}")
        lines.append("")

        for sel_key in ["sel1", "sel2"]:
            if sel_key not in results[model]:
                continue
            sel_label = SEL_LABELS.get(sel_key, sel_key)
            lines.append(f"### {sel_label}")
            lines.append("")

            for task in sorted(results[model][sel_key]):
                lines.append(f"#### `{task}`")
                lines.append("")
                task_data = results[model][sel_key][task]

                metrics = [
                    (k, METRIC_LABELS.get(k, k))
                    for k in TASK_METRICS.get(task, list(METRIC_LABELS.keys()))
                ]
                for metric_key, metric_label in metrics:
                    lines.append(f"**{metric_label}**")
                    lines.append("")

                    # Collect per-col_key data first to determine num_runs
                    col_data: dict = {}  # col_key -> (rounded_runs, mean)
                    max_runs = 0
                    for col_key in COL_KEYS:
                        raw = task_data.get(col_key, {}).get(metric_key)
                        if raw:
                            rounded, mean = compute_rows(raw)
                            col_data[col_key] = (rounded, mean)
                            max_runs = max(max_runs, len(rounded))

                    # Table header: promote ratio | Run 1 | Run 2 | ... | Mean
                    run_headers = [f"Run {i + 1}" for i in range(max_runs)]
                    header = "| Promote Rate | " + " | ".join(run_headers) + " | **Mean** |"
                    sep = "| :--- |" + " :---: |" * (max_runs + 1)
                    lines += [header, sep]

                    # One row per promote ratio
                    for col_key, col_header in zip(COL_KEYS, COL_HEADERS):
                        entry = col_data.get(col_key)
                        if entry:
                            run_cells = [fmt(v) for v in entry[0]]
                            # Pad if fewer runs than max
                            run_cells += ["--"] * (max_runs - len(run_cells))
                            mean_cell = fmt(entry[1])
                        else:
                            run_cells = ["--"] * max_runs
                            mean_cell = "--"
                        lines.append(
                            f"| {col_header} | " + " | ".join(run_cells) + f" | **{mean_cell}** |"
                        )

                    lines.append("")

    return "\n".join(lines)


def main():
    results = collect_results()
    readme = generate_readme(results)
    OUTPUT_PATH.write_text(readme)
    print(f"Written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
