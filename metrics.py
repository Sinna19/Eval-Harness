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

    # Per-criterion view for multi-turn cases: each case is tagged with the
    # judged criteria it mainly probes (focus = consistency / policy /
    # pressure / memory), so "we fail under pressure but remember context
    # fine" is visible instead of averaged into one multi_turn number. A case
    # with several tags counts toward each of them.
    by_focus = defaultdict(lambda: {"pass": 0, "total": 0})
    for r in results:
        for tag in r.get("focus") or []:
            by_focus[tag]["total"] += 1
            if r["passed"]:
                by_focus[tag]["pass"] += 1
    focus_summary = {
        tag: {**v, "rate_pct": round(100 * v["pass"] / v["total"], 1)}
        for tag, v in by_focus.items()
    }

    return {
        "usage": usage_summary(results),
        "overall": {
            "pass": total_pass,
            "total": total_cases,
            "rate_pct": round(100 * total_pass / total_cases, 1) if total_cases else 0.0,
        },
        "by_category": category_summary,
        "by_focus": focus_summary,
    }


def _percentile(values: list, pct: float):
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return ordered[idx]


def usage_summary(results: list) -> dict | None:
    """Cost / token / latency roll-up across a run. Rows from before LiteLLM
    metering (no cost_usd field) are ignored, so old result files still load.
    Error rows are included in cost (a retried-then-failed case still spent
    tokens) but excluded from latency (they never got a real answer)."""
    metered = [r for r in results if "cost_usd" in r]
    if not metered:
        return None
    latencies = [r["sut_latency_s"] for r in metered
                 if r.get("sut_latency_s") is not None and r.get("method") != "error"]
    return {
        "cost_usd": round(sum(r["cost_usd"] for r in metered), 6),
        "sut_cost_usd": round(sum(r.get("sut_cost_usd", 0) for r in metered), 6),
        "judge_cost_usd": round(sum(r.get("judge_cost_usd", 0) for r in metered), 6),
        "llm_calls": sum(r.get("llm_calls", 0) for r in metered),
        "prompt_tokens": sum(r.get("prompt_tokens", 0) for r in metered),
        "completion_tokens": sum(r.get("completion_tokens", 0) for r in metered),
        "unpriced_calls": sum(r.get("unpriced_calls", 0) for r in metered),
        "sut_latency_p50_s": _percentile(latencies, 50),
        "sut_latency_p95_s": _percentile(latencies, 95),
        "errors": sum(1 for r in metered if r.get("method") == "error"),
    }


def print_metrics(metrics: dict) -> None:
    overall = metrics["overall"]
    print(f"\n=== EVAL REPORT ===")
    print(f"Overall: {overall['pass']}/{overall['total']} passed ({overall['rate_pct']}%)\n")
    for cat, stats in metrics["by_category"].items():
        print(f"  {cat:<12} {stats['pass']}/{stats['total']} passed ({stats['rate_pct']}%)")
        for method, mstats in stats["by_method"].items():
            print(f"      via {method:<18} {mstats['pass']}/{mstats['total']} ({mstats['rate_pct']}%)")
    if metrics.get("by_focus"):
        print("\n  Multi-turn criteria (a case can count toward several):")
        for tag, stats in sorted(metrics["by_focus"].items()):
            print(f"      {tag:<12} {stats['pass']}/{stats['total']} ({stats['rate_pct']}%)")
    usage = metrics.get("usage")
    if usage:
        print(f"\n  Cost (LiteLLM list price): ${usage['cost_usd']:.4f} "
              f"(agent ${usage['sut_cost_usd']:.4f}, judge ${usage['judge_cost_usd']:.4f}) "
              f"over {usage['llm_calls']} calls, "
              f"{usage['prompt_tokens'] + usage['completion_tokens']:,} tokens")
        if usage["unpriced_calls"]:
            print(f"      {usage['unpriced_calls']} call(s) to models without pricing data counted as $0")
        if usage["sut_latency_p50_s"] is not None:
            print(f"  Agent latency per reply: p50 {usage['sut_latency_p50_s']:.2f}s, "
                  f"p95 {usage['sut_latency_p95_s']:.2f}s")


def append_metrics_history(metrics: dict, dataset_version: str, fingerprint: str,
                            path: str = "metrics_history.json",
                            models: dict | None = None) -> None:
    """Appends one timestamped entry per harness run, tagged with the
    dataset version/fingerprint that produced it, so category pass rates
    can be compared across runs and across dataset changes over time."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_version": dataset_version,
        "dataset_fingerprint": fingerprint,
        **(models or {}),
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
