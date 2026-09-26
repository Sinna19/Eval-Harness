"""
Run this file to execute the full eval suite:

    export GROQ_API_KEY="your-key-here"
    python harness.py

It will:
  1. Send every case in test_cases.py through the support agent (llm_task.py)
  2. Score each reply (scoring.py)
  3. Print a per-category, per-scoring-method pass rate summary (metrics.py)
  4. Save results.json/results.csv, tagged with the dataset version + content
     fingerprint (test_cases.py) so regression.py knows exactly what dataset
     a given result belongs to
  5. Append one entry to metrics_history.json so category pass rates are
     trackable across runs over time, not just visible one run at a time
"""

import csv
import json
import time

from llm_task import support_agent_reply
from test_cases import TEST_CASES, DATASET_VERSION, dataset_fingerprint
from scoring import score_case
from metrics import compute_metrics, print_metrics, append_metrics_history


def run_cases(cases: list) -> list:
    """Runs any list of case dicts (the full TEST_CASES suite, or any other
    list shaped the same way -- e.g. generated variants from
    generate_adversarial.py) through the SUT + scorer. This is the actual
    execution path referred to as 'the harness' elsewhere in this repo:
    factored out so other tooling can run cases through the exact same
    pipeline instead of re-implementing it."""
    results = []

        for case in cases:
        print(f"Running case: {case['id']} ({case['category']})...")
        try:
            try:
                reply = support_agent_reply(case["input"])
            except Exception as e:
                results.append({
                    "id": case["id"],
                    "category": case["category"],
                    "input": case["input"],
                    "reply": f"[ERROR calling model: {e}]",
                    "passed": False,
                    "method": "error",
                    "reason": str(e),
                })
                continue

            try:
                score = score_case(case, reply)
            except Exception as e:
                results.append({
                    "id": case["id"],
                    "category": case["category"],
                    "input": case["input"],
                    "reply": reply,
                    "passed": False,
                    "method": "error",
                    "reason": f"Scoring failed: {e}",
                })
                continue

            results.append({
                "id": case["id"],
                "category": case["category"],
                "input": case["input"],
                "reply": reply,
                "passed": score["passed"],
                "method": score["method"],
                "reason": score["reason"],
            })
        finally:
            # Always pause between cases -- including after a failure or a
            # rate-limit hit -- so hitting one 429 doesn't cause the rest of
            # the run to fire with zero delay and compound the problem.
            time.sleep(3)

    return results


def run_suite():
    """Runs the curated suite specifically (TEST_CASES). Kept as its own
    function -- rather than inlining run_cases(TEST_CASES) everywhere --
    since stability_check.py and regression.py both import run_suite()
    by name."""
    return run_cases(TEST_CASES)


def summarize(results):
    """Computes and prints the metrics summary, then the failure list.
    Kept under its original name for backward compatibility with anything
    importing it; now returns the metrics dict instead of nothing, so a
    caller (e.g. regression.py) can reuse it without recomputing."""
    metrics = compute_metrics(results)
    print_metrics(metrics)

    print("\nFailures:")
    failures = [r for r in results if not r["passed"]]
    if not failures:
        print("  None!")
    for f in failures:
        print(f"  - {f['id']}: {f['reason']}")

    return metrics


def save_results(results):
    meta = {
        "dataset_version": DATASET_VERSION,
        "dataset_fingerprint": dataset_fingerprint(),
    }
    metrics = compute_metrics(results)

    with open("results.json", "w", encoding="utf-8") as f:
        # ensure_ascii=False so smart quotes/dashes in model replies write as
        # actual UTF-8 characters instead of \uXXXX escape codes -- the file
        # is already opened with encoding="utf-8" so this is safe to read back.
        json.dump({"meta": meta, "results": results}, f, indent=2, ensure_ascii=False)

    with open("results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    append_metrics_history(metrics, meta["dataset_version"], meta["dataset_fingerprint"])

    print("\nSaved results.json and results.csv (dataset version "
          f"{meta['dataset_version']} / {meta['dataset_fingerprint']}), "
          "appended a summary to metrics_history.json")


if __name__ == "__main__":
    all_results = run_suite()
    summarize(all_results)
    save_results(all_results)
