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

Every result row also carries what the case cost (LiteLLM list price),
tokens used, and the agent's latency, and results.json records which agent
and judge models produced it. Pick models without code changes:

    python harness.py --sut-model groq/openai/gpt-oss-20b --out runs/gpt-oss-20b
    python harness.py --cases normal_1,mt_authority_audit_pressure
"""

import argparse
import csv
import json
import os
import sys
import time
from importlib.metadata import version, PackageNotFoundError

import llm_task
from llm_task import support_agent_reply, support_agent_chat_reply, track_usage
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


def _usage_fields(usage: dict) -> dict:
    """Flat per-row usage columns (same keys on every row, see _row)."""
    sut, judge = usage["by_role"]["sut"], usage["by_role"]["judge"]
    return {
        "llm_calls": usage["calls"],
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "cost_usd": round(usage["cost_usd"], 6),
        "sut_cost_usd": round(sut["cost_usd"], 6),
        "judge_cost_usd": round(judge["cost_usd"], 6),
        # Mean latency of the agent's calls in this case (one per turn), so
        # long multi-turn cases aren't penalized just for having more turns.
        "sut_latency_s": round(sut["latency_s"] / sut["calls"], 3) if sut["calls"] else None,
        "unpriced_calls": usage["unpriced_calls"],
    }


def run_one_case(case: dict) -> dict:
    """Runs and scores a single case (either shape) and returns its row,
    including what the case cost and how fast the agent answered."""
    with track_usage() as usage:
        row = _run_and_score(case)
    row.update(_usage_fields(usage))
    return row


def _run_and_score(case: dict) -> dict:
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


def run_metadata() -> dict:
    """What produced a results file: dataset AND models, so two results
    files can't be silently compared across different agents or judges."""
    try:
        litellm_version = version("litellm")
    except PackageNotFoundError:
        litellm_version = None
    return {
        "dataset_version": DATASET_VERSION,
        "dataset_fingerprint": dataset_fingerprint(),
        "sut_model": llm_task.SUT_MODEL,
        "judge_model": llm_task.JUDGE_MODEL,
        "litellm_version": litellm_version,
    }


def save_results(results, out_dir: str = "."):
    meta = run_metadata()
    metrics = compute_metrics(results)
    os.makedirs(out_dir, exist_ok=True)
    results_json = os.path.join(out_dir, "results.json")
    results_csv = os.path.join(out_dir, "results.csv")
    history = os.path.join(out_dir, "metrics_history.json")

    with open(results_json, "w", encoding="utf-8") as f:
        # ensure_ascii=False so smart quotes/dashes in model replies write as
        # actual UTF-8 characters instead of \uXXXX escape codes -- the file
        # is already opened with encoding="utf-8" so this is safe to read back.
        json.dump({"meta": meta, "results": results}, f, indent=2, ensure_ascii=False)

    with open(results_csv, "w", newline="", encoding="utf-8") as f:
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

    append_metrics_history(metrics, meta["dataset_version"], meta["dataset_fingerprint"],
                           path=history, models={"sut_model": meta["sut_model"],
                                                 "judge_model": meta["judge_model"]})

    print(f"\nSaved {results_json} and {results_csv} (dataset version "
          f"{meta['dataset_version']} / {meta['dataset_fingerprint']}, "
          f"agent {meta['sut_model']}, judge {meta['judge_model']}), "
          f"appended a summary to {history}")


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Run the eval suite.")
    ap.add_argument("--sut-model", help="LiteLLM model string for the agent (overrides SUT_MODEL)")
    ap.add_argument("--judge-model", help="LiteLLM model string for the judge (overrides JUDGE_MODEL)")
    ap.add_argument("--cases", help="comma-separated case ids to run instead of the full suite")
    ap.add_argument("--out", default=".", help="directory for results.json/csv and metrics_history.json")
    return ap.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    # Models are read at import time by llm_task and scoring, so CLI
    # overrides are applied by re-running this script with the env var set.
    # Keeps one source of truth (the env) instead of patching module globals.
    # (subprocess rather than os.execv, which misbehaves on Windows consoles.)
    overrides = {k: v for k, v in (("SUT_MODEL", args.sut_model),
                                   ("JUDGE_MODEL", args.judge_model)) if v}
    if any(os.environ.get(k) != v for k, v in overrides.items()):
        import subprocess
        sys.exit(subprocess.run([sys.executable, *sys.argv],
                                env={**os.environ, **overrides}).returncode)

    if llm_task.SUT_MODEL == llm_task.JUDGE_MODEL:
        print(f"WARNING: agent and judge are the same model ({llm_task.SUT_MODEL}); "
              "expect self-preferencing bias in judged cases.")

    cases = TEST_CASES
    if args.cases:
        wanted = [c.strip() for c in args.cases.split(",") if c.strip()]
        unknown = sorted(set(wanted) - {c["id"] for c in TEST_CASES})
        if unknown:
            sys.exit(f"unknown case id(s): {unknown}")
        cases = [c for c in TEST_CASES if c["id"] in wanted]

    all_results = run_cases(cases)
    summarize(all_results)
    save_results(all_results, out_dir=args.out)
