"""
Re-run a few chosen cases several times, with full replies and judge reasons.
Much cheaper than the whole suite -- use it while tuning policy or test cases.

    python run_subset.py 3 normal_1 mt_authority_audit_pressure

Prints PASS/FAIL per run; for failures also the judge's reason and the reply.
Writes subset_log.json (every run's reply, reason and full transcript).
Rows with method "error" are API problems, not agent failures.
"""

import json
import sys

from harness import run_cases
from test_cases import TEST_CASES


def main():
    if len(sys.argv) < 3:
        sys.exit("usage: python run_subset.py <runs> <case_id> [<case_id> ...]")
    runs, ids = int(sys.argv[1]), sys.argv[2:]
    known = {c["id"] for c in TEST_CASES}
    unknown = [i for i in ids if i not in known]
    if unknown:
        sys.exit(f"unknown case id(s): {unknown}")
    cases = [c for c in TEST_CASES if c["id"] in ids]

    log = []
    for i in range(runs):
        for r in run_cases(cases):
            print(f"run {i + 1}  {r['id']:<34} {'PASS' if r['passed'] else 'FAIL'}  [{r['method']}]")
            if not r["passed"]:
                print("    reason:", r["reason"][:300])
                print("    reply :", r["reply"][:300].replace("\n", " "))
            log.append({"run": i + 1, **{k: r[k] for k in
                        ("id", "passed", "method", "reason", "reply", "transcript")}})
    with open("subset_log.json", "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print("\nSaved subset_log.json")


if __name__ == "__main__":
    main()
