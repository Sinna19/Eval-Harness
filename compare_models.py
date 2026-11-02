"""
Compare several agent models on the same suite, with the SAME judge, and
get one table: pass rate, errors, cost and latency per model.

    python compare_models.py groq/openai/gpt-oss-20b groq/openai/gpt-oss-120b
    python compare_models.py groq/openai/gpt-oss-120b gemini/gemini-2.5-flash \
        --cases normal_1,mt_authority_audit_pressure      # cheap trial first

Each model runs through `harness.py --sut-model <model> --out runs/<slug>`,
i.e. the exact pipeline every other tool here uses; nothing is
re-implemented. Results land in runs/<slug>/results.json and the table is
written to runs/comparison.md.

Keeping the judge fixed is the point: if the judge changed between rows,
a pass-rate difference could be the grader, not the agent. A model that is
also the judge is refused for the same reason (self-preferencing bias).

`--report-only` rebuilds the table from existing runs/ without calling any
model, e.g. after a run was interrupted by a free-tier daily limit.
"""

import argparse
import json
import os
import re
import subprocess
import sys

from llm_task import JUDGE_MODEL, resolve_model


def slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("_")


def run_model(model: str, out_dir: str, cases: str | None, judge: str) -> int:
    cmd = [sys.executable, "harness.py", "--sut-model", model,
           "--judge-model", judge, "--out", out_dir]
    if cases:
        cmd += ["--cases", cases]
    print(f"\n######## {model} -> {out_dir}")
    return subprocess.run(cmd).returncode


def load_row(model: str, out_dir: str) -> dict | None:
    path = os.path.join(out_dir, "results.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # Imported lazily so --help works without the dataset modules.
    from metrics import compute_metrics
    m = compute_metrics(data["results"])
    u = m.get("usage") or {}
    n = m["overall"]["total"]
    graded = n - u.get("errors", 0)
    return {
        "model": model,
        "judge": data["meta"].get("judge_model"),
        "pass": m["overall"]["pass"],
        "total": n,
        # Pass rate over cases that actually got an answer: an API error is
        # not a policy failure (same convention as stability_check.py).
        "rate_graded": round(100 * m["overall"]["pass"] / graded, 1) if graded else None,
        "errors": u.get("errors"),
        "cost_usd": u.get("cost_usd"),
        "sut_cost_usd": u.get("sut_cost_usd"),
        "unpriced_calls": u.get("unpriced_calls", 0),
        "p50": u.get("sut_latency_p50_s"),
        "p95": u.get("sut_latency_p95_s"),
        "dataset": f"{data['meta'].get('dataset_version')} / {data['meta'].get('dataset_fingerprint')}",
    }


def fmt(v, spec="", none="n/a"):
    return none if v is None else format(v, spec)


def render(rows: list) -> str:
    judges = {r["judge"] for r in rows}
    datasets = {r["dataset"] for r in rows}
    lines = [
        "# Model comparison",
        "",
        f"Judge: `{', '.join(sorted(judges))}`" + ("  **(WARNING: judges differ between rows)**" if len(judges) > 1 else ""),
        f"Dataset: {', '.join(sorted(datasets))}" + ("  **(WARNING: datasets differ)**" if len(datasets) > 1 else ""),
        "",
        "| Agent model | Passed | Pass rate (graded) | API errors | Agent cost | Total cost | Latency p50 / p95 |",
        "|---|:-:|:-:|:-:|--:|--:|:-:|",
    ]
    for r in rows:
        unpriced = "*" if r["unpriced_calls"] else ""
        lines.append(
            f"| `{r['model']}` | {r['pass']}/{r['total']} | {fmt(r['rate_graded'])}% | {fmt(r['errors'])} "
            f"| ${fmt(r['sut_cost_usd'], '.4f')}{unpriced} | ${fmt(r['cost_usd'], '.4f')}{unpriced} "
            f"| {fmt(r['p50'], '.2f')}s / {fmt(r['p95'], '.2f')}s |"
        )
    lines += [
        "",
        "Costs are LiteLLM list prices for the tokens used; on free tiers nothing is billed, "
        "but the number shows what the same run would cost at scale."
        + (" `*` = some calls went to models without pricing data and count as $0." if any(r["unpriced_calls"] for r in rows) else ""),
        "Latency is the agent's time per reply (judge calls excluded); error rows are excluded.",
    ]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+", help="LiteLLM model strings for the agent")
    ap.add_argument("--judge-model", default=JUDGE_MODEL, help=f"fixed judge for every row (default {JUDGE_MODEL})")
    ap.add_argument("--cases", help="comma-separated case ids (default: full suite)")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--report-only", action="store_true", help="only rebuild the table from existing runs")
    args = ap.parse_args()

    judge = resolve_model(args.judge_model)
    models = [resolve_model(m) for m in args.models]
    if judge in models:
        sys.exit(f"{judge} is both an agent under test and the judge; pick a different --judge-model.")

    rows = []
    for model in models:
        out_dir = os.path.join(args.runs_dir, slug(model))
        if not args.report_only:
            code = run_model(model, out_dir, args.cases, judge)
            if code != 0:
                print(f"!! harness exited with {code} for {model}; keeping whatever it saved")
        row = load_row(model, out_dir)
        if row is None:
            print(f"!! no results for {model} in {out_dir}, skipping")
            continue
        rows.append(row)

    if not rows:
        sys.exit("No results to compare.")
    table = render(rows)
    os.makedirs(args.runs_dir, exist_ok=True)
    out = os.path.join(args.runs_dir, "comparison.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(table)
    print("\n" + table + f"\nSaved {out}")


if __name__ == "__main__":
    main()
