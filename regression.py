"""
Regression gate: runs the suite fresh and compares it against a saved
baseline (baseline_results.json by default), flagging any case that PASSED
in the baseline but FAILS now as a regression -- distinct from a case that
was already failing, or one that's new to the dataset since the baseline
was saved.

Deliberately compares against an explicitly-saved baseline, not "the
previous run": the previous run silently drifts every time you run the
suite, and individual runs are already known to be noisy at temperature=0.3
(see stability_check.py / README). A baseline is something you promote on
purpose once you're satisfied a given state of the policy + suite is good --
the same reason you wouldn't want a CI gate to quietly redefine "correct" as
"whatever happened to run last."

Each baseline is tagged with the dataset version/fingerprint (test_cases.py)
that produced it, so if the dataset has changed since, that's surfaced
explicitly rather than silently comparing apples to oranges.

Usage:
    export GROQ_API_KEY="your-key-here"
    python regression.py                    # run suite, compare to baseline, exit 1 on regression
    python regression.py --update-baseline  # run suite, save it AS the new baseline instead
    python regression.py --baseline other.json
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

from harness import run_suite
from test_cases import DATASET_VERSION, dataset_fingerprint

DEFAULT_BASELINE_PATH = "baseline_results.json"


def load_baseline(path: str = DEFAULT_BASELINE_PATH):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_baseline(results: list, path: str = DEFAULT_BASELINE_PATH) -> None:
    baseline = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "dataset_version": DATASET_VERSION,
        "dataset_fingerprint": dataset_fingerprint(),
        "results": {r["id"]: r["passed"] for r in results},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2, ensure_ascii=False)
    print(f"Saved baseline ({len(results)} cases) to {path}")


def compare_to_baseline(results: list, baseline: dict) -> dict:
    current = {r["id"]: r["passed"] for r in results}
    base_results = baseline.get("results", {})

    regressions, fixed, new_cases, still_failing, still_passing = [], [], [], [], []

    for case_id, passed_now in current.items():
        if case_id not in base_results:
            new_cases.append(case_id)
            continue
        passed_before = base_results[case_id]
        if passed_before and not passed_now:
            regressions.append(case_id)
        elif not passed_before and passed_now:
            fixed.append(case_id)
        elif passed_now:
            still_passing.append(case_id)
        else:
            still_failing.append(case_id)

    removed_cases = [cid for cid in base_results if cid not in current]

    return {
        "regressions": regressions,
        "fixed": fixed,
        "new_cases": new_cases,
        "removed_cases": removed_cases,
        "still_failing": still_failing,
        "still_passing": still_passing,
    }


def print_comparison(comparison: dict, baseline: dict) -> None:
    print("\n=== REGRESSION CHECK ===")
    print(f"Baseline dataset version: {baseline.get('dataset_version', 'unknown')} "
          f"(fingerprint {baseline.get('dataset_fingerprint', 'unknown')})")
    print(f"Current dataset version:  {DATASET_VERSION} (fingerprint {dataset_fingerprint()})")
    if baseline.get("dataset_fingerprint") != dataset_fingerprint():
        print("NOTE: the dataset has changed since this baseline was saved -- the "
              "case-by-case comparison below only covers cases present in both.")

    if comparison["regressions"]:
        print(f"\nREGRESSIONS ({len(comparison['regressions'])}) -- passed in baseline, failing now:")
        for cid in comparison["regressions"]:
            print(f"  \u2717 {cid}")
    else:
        print("\nNo regressions -- nothing that passed in the baseline is now failing.")

    if comparison["fixed"]:
        print(f"\nFixed since baseline ({len(comparison['fixed'])}):")
        for cid in comparison["fixed"]:
            print(f"  \u2713 {cid}")

    if comparison["new_cases"]:
        print(f"\nNew cases (no baseline entry to compare against): {comparison['new_cases']}")
    if comparison["removed_cases"]:
        print(f"\nCases in baseline no longer in the suite: {comparison['removed_cases']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default=DEFAULT_BASELINE_PATH,
                         help=f"Path to baseline file (default: {DEFAULT_BASELINE_PATH})")
    parser.add_argument("--update-baseline", action="store_true",
                         help="Run the suite and save the result AS the new baseline, instead of comparing.")
    args = parser.parse_args()

    print("Running suite for regression check...")
    results = run_suite()

    if args.update_baseline:
        save_baseline(results, path=args.baseline)
        return

    baseline = load_baseline(args.baseline)
    if baseline is None:
        print(f"No baseline found at {args.baseline}. Run with --update-baseline first.")
        sys.exit(1)

    comparison = compare_to_baseline(results, baseline)
    print_comparison(comparison, baseline)

    if comparison["regressions"]:
        print(f"\nFAIL: {len(comparison['regressions'])} regression(s) found.")
        sys.exit(1)

    print("\nOK: no regressions.")
    sys.exit(0)


if __name__ == "__main__":
    main()
