"""
Run this file to execute the full eval suite:

    export GEMINI_API_KEY="your-key-here"
    python harness.py

It will:
  1. Send every case in test_cases.py through the support agent (llm_task.py)
  2. Score each reply (scoring.py)
  3. Print a per-category pass rate summary
  4. Save full results to results.csv and results.json for your resume/portfolio
"""

import csv
import json
import time
from collections import defaultdict

from llm_task import support_agent_reply
from test_cases import TEST_CASES
from scoring import score_case


def run_suite():
    results = []

    for case in TEST_CASES:
        print(f"Running case: {case['id']} ({case['category']})...")
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

        score = score_case(case, reply)
        results.append({
            "id": case["id"],
            "category": case["category"],
            "input": case["input"],
            "reply": reply,
            "passed": score["passed"],
            "method": score["method"],
            "reason": score["reason"],
        })

        # Small delay between cases to be a good citizen of the free tier.
        time.sleep(2)

    return results


def summarize(results):
    by_category = defaultdict(lambda: {"pass": 0, "total": 0})
    for r in results:
        by_category[r["category"]]["total"] += 1
        if r["passed"]:
            by_category[r["category"]]["pass"] += 1

    print("\n=== EVAL REPORT ===")
    total_pass = sum(v["pass"] for v in by_category.values())
    total_cases = sum(v["total"] for v in by_category.values())
    print(f"Overall: {total_pass}/{total_cases} passed "
          f"({100 * total_pass / total_cases:.0f}%)\n")

    for category, counts in by_category.items():
        rate = 100 * counts["pass"] / counts["total"]
        print(f"  {category:<12} {counts['pass']}/{counts['total']} passed ({rate:.0f}%)")

    print("\nFailures:")
    failures = [r for r in results if not r["passed"]]
    if not failures:
        print("  None!")
    for f in failures:
        print(f"  - {f['id']}: {f['reason']}")


def save_results(results):
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open("results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print("\nSaved results.json and results.csv")


if __name__ == "__main__":
    all_results = run_suite()
    summarize(all_results)
    save_results(all_results)
