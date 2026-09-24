"""
Richer eval metrics than harness.py's original per-category pass rate:
also breaks each category down by scoring method (rule_based / llm_judge /
error), and appends every run to metrics_history.json so pass rates can be
tracked across runs and dataset versions over time -- not just whatever the
most recent run happened to print.
"""

import json
import os
from collections import defaultdict
from datetime import datetime, timezone


def compute_metrics(results: list) -> dict:
    """Per-category stats, further broken down by scoring method, plus an
    overall summary. Method breakdown matters because a category's pass
    rate alone can hide, e.g., all its failures coming from the 'error'
    method (rate-limit/API issues) rather than real policy violations."""
    by_category = defaultdict(lambda: {
        "pass": 0, "total": 0,
        "by_method": defaultdict(lambda: {"pass": 0, "total": 0}),
    })

    for r in results:
        cat = by_category[r["category"]]
        cat["total"] += 1
        method_stats = cat["by_method"][r.get("method", "unknown")]
        method_stats["total"] += 1
        if r["passed"]:
            cat["pass"] += 1
            method_stats["pass"] += 1

    category_summary = {}
    for cat, stats in by_category.items():
        category_summary[cat] = {
            "pass": stats["pass"],
            "total": stats["total"],
            "rate_pct": round(100 * stats["pass"] / stats["total"], 1) if stats["total"] else 0.0,
            "by_method": {
                m: {
                    "pass": ms["pass"],
                    "total": ms["total"],
                    "rate_pct": round(100 * ms["pass"] / ms["total"], 1) if ms["total"] else 0.0,
                }
                for m, ms in stats["by_method"].items()
            },
        }

    total_pass = sum(v["pass"] for v in category_summary.values())
    total_cases = sum(v["total"] for v in category_summary.values())

    return {
        "overall": {
            "pass": total_pass,
            "total": total_cases,
            "rate_pct": round(100 * total_pass / total_cases, 1) if total_cases else 0.0,
        },
        "by_category": category_summary,
    }


def print_metrics(metrics: dict) -> None:
    overall = metrics["overall"]
    print(f"\n=== EVAL REPORT ===")
    print(f"Overall: {overall['pass']}/{overall['total']} passed ({overall['rate_pct']}%)\n")
    for cat, stats in metrics["by_category"].items():
        print(f"  {cat:<12} {stats['pass']}/{stats['total']} passed ({stats['rate_pct']}%)")
        for method, mstats in stats["by_method"].items():
            print(f"      via {method:<10} {mstats['pass']}/{mstats['total']} ({mstats['rate_pct']}%)")


def append_metrics_history(metrics: dict, dataset_version: str, fingerprint: str,
                            path: str = "metrics_history.json") -> None:
    """Appends one timestamped entry per harness run, tagged with the
    dataset version/fingerprint that produced it, so category pass rates
    can be compared across runs and across dataset changes over time."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_version": dataset_version,
        "dataset_fingerprint": fingerprint,
        "metrics": metrics,
    }

    history = []
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                history = json.load(f)
        except (json.JSONDecodeError, OSError):
            history = []

    history.append(entry)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
