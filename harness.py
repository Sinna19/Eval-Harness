"""
Run this file to execute the full eval suite:

    export GROQ_API_KEY="your-key-here"
    python harness.py

It will:
  1. Send every case in test_cases.py through the support agent (llm_task.py).
     Single-turn cases are one message in, one reply out; multi-turn cases
     (those with a `turns` list) are played turn by turn with the full
     history replayed to the agent each time
  2. Score each reply -- or, for multi-turn cases, the WHOLE transcript
     (scoring.py)
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

from llm_task import support_agent_reply, support_agent_chat_reply
from test_cases import TEST_CASES, DATASET_VERSION, dataset_fingerprint, is_conversation
from scoring import score_case, score_conversation
from metrics import compute_metrics, print_metrics, append_metrics_history


TURN_DELAY_SECONDS = 8   # pause between turns of one conversation
CASE_DELAY_SECONDS = 10   # pause between cases


def _row(case: dict, *, reply: str, transcript: list, passed: bool, method: str,
         reason: str, failed_turn=None, judge_flag=None) -> dict:
    """One result row. Every row -- single- or multi-turn -- has the SAME
    keys, so results.csv (whose header is taken from the first row) can't
    break when a later row has extra fields."""
    return {
        "id": case["id"],
        "category": case["category"],
        "input": case["input"],
        "reply": reply,
        "passed": passed,
        "method": method,
        "reason": reason,
        "turns": sum(1 for m in transcript if m["role"] == "customer"),
        "failed_turn": failed_turn,
        "judge_flag": judge_flag,
        "focus": case.get("focus", []),
        "transcript": transcript,
    }


def play_conversation(case: dict) -> tuple[list, str | None]:
    """Plays a multi-turn case. Returns (transcript, error). The agent is
    stateless, so each turn replays the full history so far -- that replay
    is the only thing giving it memory of earlier turns. If the model call
    fails mid-conversation, the partial transcript is returned with the
    error text so the failure row still shows how far it got."""
    transcript, history = [], []
    for i, customer_msg in enumerate(case["turns"]):
        history.append({"role": "user", "content": customer_msg})
        transcript.append({"role": "customer", "content": customer_msg})
        try:
            reply = support_agent_chat_reply(history)
        except Exception as e:
            return transcript, f"turn {i + 1}: {e}"
        history.append({"role": "assistant", "content": reply})
        transcript.append({"role": "ai", "content": reply})
        if i < len(case["turns"]) - 1:
            time.sleep(TURN_DELAY_SECONDS)
    return transcript, None


def run_one_case(case: dict) -> dict:
    """Runs and scores a single case (either shape) and returns its row."""
    if is_conversation(case):
        transcript, err = play_conversation(case)
        last_reply = next((m["content"] for m in reversed(transcript) if m["role"] == "ai"), "")
        if err:
            return _row(case, reply=f"[ERROR calling model: {err}]", transcript=transcript,
                        passed=False, method="error", reason=err)
        try:
            score = score_conversation(case, transcript)
        except Exception as e:
            return _row(case, reply=last_reply, transcript=transcript, passed=False,
                        method="error", reason=f"Scoring failed: {e}")
        return _row(case, reply=last_reply, transcript=transcript, passed=score["passed"],
                    method=score["method"], reason=score["reason"],
                    failed_turn=score.get("failed_turn"), judge_flag=score.get("judge_flag"))

    transcript = [{"role": "customer", "content": case["input"]}]
    try:
        reply = support_agent_reply(case["input"])
    except Exception as e:
        return _row(case, reply=f"[ERROR calling model: {e}]", transcript=transcript,
                    passed=False, method="error", reason=str(e))
    transcript.append({"role": "ai", "content": reply})
    try:
        score = score_case(case, reply)
    except Exception as e:
        return _row(case, reply=reply, transcript=transcript, passed=False,
                    method="error", reason=f"Scoring failed: {e}")
    return _row(case, reply=reply, transcript=transcript, passed=score["passed"],
                method=score["method"], reason=score["reason"])


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
            results.append(run_one_case(case))
        finally:
            # Always pause between cases -- including after a failure or a
            # rate-limit hit -- so hitting one 429 doesn't cause the rest of
            # the run to fire with zero delay and compound the problem.
            time.sleep(CASE_DELAY_SECONDS)

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
        where = f" [broke at turn {f['failed_turn']}]" if f.get("failed_turn") else ""
        print(f"  - {f['id']}{where}: {f['reason']}")

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
        # Union of keys across ALL rows (not just row 0), and list/dict
        # values (transcript, focus) JSON-encoded so each fits one cell.
        fieldnames = list(dict.fromkeys(k for r in results for k in r))
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        for r in results:
            writer.writerow({
                k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                for k, v in r.items()
            })

    append_metrics_history(metrics, meta["dataset_version"], meta["dataset_fingerprint"])

    print("\nSaved results.json and results.csv (dataset version "
          f"{meta['dataset_version']} / {meta['dataset_fingerprint']}), "
          "appended a summary to metrics_history.json")


if __name__ == "__main__":
    all_results = run_suite()
    summarize(all_results)
    save_results(all_results)
