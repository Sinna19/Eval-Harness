"""
Runs the full eval suite N times (default 5) to check how STABLE each
case's pass/fail result is under temperature=0.3 sampling -- a case that
passes once and fails once with identical input isn't "fixed," it's a
coin flip. This complements harness.py (a single run) rather than
replacing it: harness.py tells you whether the suite passes right now,
this tells you whether that result is one you can trust.

Usage:
    export GROQ_API_KEY="your-key-here"
    python stability_check.py            # 5 runs (default)
    python stability_check.py --runs 10  # override run count

Writes stability_report.json with the full raw data (every run, every
case) and prints a per-case summary table to the console. Any case that
isn't 100% consistent across runs is called out explicitly at the end --
those are the ones worth a closer look, same way edge_ambiguous_refund
was caught by comparing two individual runs by hand.
"""

import argparse
import json
import time
from collections import defaultdict

from harness import run_suite
from test_cases import TEST_CASES


def run_multiple(n_runs: int):
    """Runs the suite n_runs times, returns a list of result-lists (one per run).
    Saves a partial report after every run -- found for real: a Groq daily
    token cap crashed run 4/5 and, before this, would have discarded runs
    1-3's data since nothing was written until the very end."""
    all_runs = []
    for run_num in range(1, n_runs + 1):
        print(f"\n{'=' * 50}")
        print(f"STABILITY CHECK -- run {run_num}/{n_runs}")
        print(f"{'=' * 50}")
        results = run_suite()
        all_runs.append(results)

        case_stats = aggregate(all_runs)
        save_report(all_runs, case_stats, n_runs, path="stability_report.json")
        print(f"(Progress saved: {run_num}/{n_runs} runs so far)")

        # Small pause between full runs, on top of the per-case delay
        # already inside run_suite(), to stay easy on the free tier.
        if run_num < n_runs:
            time.sleep(5)

    return all_runs


def aggregate(all_runs):
    """
    Builds a per-case stability summary:
      { case_id: {"passes": int, "total": int, "outcomes": [bool, ...]} }
    """
    by_case = defaultdict(lambda: {"passes": 0, "total": 0, "outcomes": []})

    for run_results in all_runs:
        for r in run_results:
            entry = by_case[r["id"]]
            entry["total"] += 1
            entry["outcomes"].append(r["passed"])
            if r["passed"]:
                entry["passes"] += 1

    return by_case


def print_summary(by_case, n_runs):
    print(f"\n{'=' * 60}")
    print(f"STABILITY SUMMARY across {n_runs} runs")
    print(f"{'=' * 60}\n")

    # Keep the original test_cases.py ordering for readability.
    case_order = [c["id"] for c in TEST_CASES]

    unstable = []
    for case_id in case_order:
        stats = by_case.get(case_id)
        if not stats:
            continue
        rate = 100 * stats["passes"] / stats["total"]
        marker = "  " if rate in (0, 100) else "⚠️ "
        print(f"{marker}{case_id:<28} {stats['passes']}/{stats['total']} passed ({rate:.0f}%)")
        if rate not in (0, 100):
            unstable.append((case_id, stats))

    print()
    if unstable:
        print(f"{len(unstable)} case(s) gave DIFFERENT results across identical runs "
              f"-- these are not reliably fixed, they're flaky:")
        for case_id, stats in unstable:
            print(f"  - {case_id}: {stats['outcomes']}")
        print("\nConsider this a signal to tighten the policy or the test case itself, "
              "not just re-running until it happens to pass.")
    else:
        print("All cases gave the SAME result on every run. "
              "No flakiness detected across this sample size.")
        print("(Note: this doesn't prove zero flakiness -- only that none showed up "
              "in this many runs. A rare flip could still exist.)")


def save_report(all_runs, by_case, target_runs, path="stability_report.json"):
    report = {
        "target_runs": target_runs,
        "completed_runs": len(all_runs),
        "per_case_summary": {
            case_id: {
                "passes": stats["passes"],
                "total": stats["total"],
                "pass_rate_pct": round(100 * stats["passes"] / stats["total"], 1),
                "outcomes": stats["outcomes"],
            }
            for case_id, stats in by_case.items()
        },
        "raw_runs": all_runs,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nSaved full details to {path} ({len(all_runs)}/{target_runs} runs completed)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5,
                         help="Number of times to run the full suite (default: 5)")
    args = parser.parse_args()

    if args.runs < 2:
        raise SystemExit("--runs must be at least 2 to check stability across repeats.")

    runs = run_multiple(args.runs)
    case_stats = aggregate(runs)
    print_summary(case_stats, args.runs)
    save_report(runs, case_stats, args.runs)
