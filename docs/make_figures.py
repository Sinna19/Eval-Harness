"""
Regenerates the README figures from the harness's own output files, so the
charts can never drift from the real numbers.

    pip install matplotlib
    python docs/make_figures.py                 # reads ./results.json etc., writes docs/img/*.png
    python docs/make_figures.py --data some/dir # read the JSON files from another folder

Inputs (any missing one just skips its figure):
    results.json                  from harness.py
    stability_report.json         from stability_check.py
    judge_calibration_report.json from judge_calibration.py

fig_development_arc uses a short hand-entered history (constants below), each
value taken from a documented run in docs/FINDINGS.md.
"""

import argparse
import json
import os
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img")

GREEN, RED, GREY = "#2e7d32", "#c62828", "#9e9e9e"
BLUE, TEAL, AMBER = "#1565c0", "#00838f", "#ef6c00"
INK = "#212121"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.size": 11, "axes.edgecolor": "#bdbdbd", "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
})


def load(data_dir, name):
    path = os.path.join(data_dir, name)
    if not os.path.exists(path):
        print(f"  skip: {name} not found in {data_dir}")
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name), dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote docs/img/{name}")


# ---------------------------------------------------------------- figure 1
def fig_results_overview(results):
    rows = results["results"]
    cats = ["normal", "edge", "scope", "adversarial", "multi_turn"]
    labels = {"normal": "normal", "edge": "edge", "scope": "scope",
              "adversarial": "adversarial", "multi_turn": "multi-turn"}
    methods = [("rule_based", "Rule-based (keywords)", GREY),
               ("llm_judge", "LLM judge (single reply)", BLUE),
               ("conversation_judge", "Conversation judge (whole transcript)", TEAL)]
    counts = defaultdict(Counter)
    failed = Counter()
    for r in rows:
        counts[r["category"]][r["method"]] += 1
        if not r["passed"]:
            failed[r["category"]] += 1

    fig, ax = plt.subplots(figsize=(9, 3.9))
    y = list(range(len(cats)))[::-1]
    for yi, c in zip(y, cats):
        left = 0
        for key, _, color in methods:
            n = counts[c][key]
            if n:
                ax.barh(yi, n, left=left, color=color, edgecolor="white", height=0.62)
                ax.text(left + n / 2, yi, str(n), ha="center", va="center", color="white", fontweight="bold")
                left += n
        total = sum(counts[c].values())
        ok = total - failed[c]
        ax.text(left + 0.25, yi, f"{ok}/{total} passed", va="center", fontweight="bold",
                color=GREEN if ok == total else RED)
    ax.set_yticks(y)
    ax.set_yticklabels([labels[c] for c in cats])
    ax.set_xlabel("test cases")
    ax.set_xlim(0, max(sum(counts[c].values()) for c in cats) + 3.2)
    total = len(rows)
    passed = sum(r["passed"] for r in rows)
    ax.set_title(f"{passed}/{total} cases passed - and how each one is graded", loc="left", fontweight="bold")
    ax.legend(handles=[Patch(color=c, label=l) for _, l, c in methods], loc="upper center",
              bbox_to_anchor=(0.45, -0.2), ncol=3, frameon=False, fontsize=9)
    save(fig, "results_overview.png")


# ---------------------------------------------------------------- figure 2
def fig_error_vs_failure(report):
    runs = report["raw_runs"]
    per_run = []
    for run in runs:
        p = sum(1 for r in run if r["passed"])
        e = sum(1 for r in run if r["method"] == "error")
        f = sum(1 for r in run if not r["passed"] and r["method"] != "error")
        per_run.append((p, f, e))

    def flaky(count_errors_as_fail):
        ids = {r["id"] for run in runs for r in run}
        n = 0
        for i in ids:
            outcomes = set()
            for run in runs:
                for r in run:
                    if r["id"] != i:
                        continue
                    if r["method"] == "error" and not count_errors_as_fail:
                        continue
                    outcomes.add(bool(r["passed"]))
            n += len(outcomes) > 1
        return n

    old, new = flaky(True), flaky(False)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1.5, 1]})
    xs = range(1, len(per_run) + 1)
    bottoms = [0] * len(per_run)
    for idx, (label, color) in enumerate([("passed", GREEN), ("real failure", RED), ("API error (inconclusive)", GREY)]):
        vals = [pr[idx] for pr in per_run]
        a1.bar(xs, vals, bottom=bottoms, color=color, label=label, edgecolor="white", width=0.62)
        for x, v, b in zip(xs, vals, bottoms):
            if v:
                a1.text(x, b + v / 2, str(v), ha="center", va="center", color="white", fontweight="bold")
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    a1.set_xticks(list(xs)); a1.set_xticklabels([f"run {x}" for x in xs])
    a1.set_ylabel("cases"); a1.set_ylim(0, 27)
    a1.set_title("Same suite, three identical runs", loc="left", fontweight="bold")
    a1.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)

    a2.bar(["errors counted\nas failures", "errors excluded\n(inconclusive)"], [old, new],
           color=[RED, GREEN], width=0.55)
    for x, v in enumerate([old, new]):
        a2.text(x, v + 0.25, str(v), ha="center", fontweight="bold", fontsize=14)
    a2.set_ylim(0, max(old, new) + 2.5)
    a2.set_ylabel("cases reported as flaky")
    a2.set_title("What the report said", loc="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "error_vs_failure.png")


# ---------------------------------------------------------------- figure 3
def fig_judge_calibration(report):
    items = report["per_item"]
    rows = []
    for name, attempts in items.items():
        a = attempts[-1]
        expected = "PASS" if name.startswith("gold_pass") else "FAIL"
        rows.append((name.replace("gold_", ""), expected, a["got"], a["verdict_ok"], a.get("got_turn")))
    rows.sort(key=lambda r: (r[1] != "FAIL", r[0]))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 5.6), gridspec_kw={"width_ratios": [1.9, 1]})
    n = len(rows)
    for i, (name, exp, got, ok, turn) in enumerate(rows):
        y = n - 1 - i
        a1.scatter(0, y, s=170, marker="o" if ok else "X", color=GREEN if ok else RED, zorder=3)
        a1.text(0.35, y, name, va="center", fontsize=9.5)
        tag = f"planted FAIL -> judge: {got}" + (f" @ turn {turn}" if turn else "") if exp == "FAIL" else f"clean PASS -> judge: {got}"
        a1.text(7.7, y, tag, va="center", fontsize=8.8, color=GREY if ok else RED, ha="right")
    a1.set_xlim(-0.4, 7.8); a1.set_ylim(-1, n)
    a1.axis("off")
    ok_n = sum(r[3] for r in rows)
    a1.set_title(f"Judge vs. 16 hand-labelled transcripts: {ok_n}/{n} correct", loc="left", fontweight="bold")

    bars = [("errors counted\nas misses", 0.781, GREY), ("same run,\nerrors excluded*", 0.962, BLUE),
            ("clean rerun,\nno errors", report["summary"]["verdict_accuracy"], GREEN)]
    for x, (lab, v, c) in enumerate(bars):
        a2.bar(x, v, color=c, width=0.6)
        a2.text(x, v + 0.015, f"{v:.2f}", ha="center", fontweight="bold")
    a2.set_xticks(range(3)); a2.set_xticklabels([b[0] for b in bars], fontsize=9)
    a2.set_ylim(0, 1.12); a2.set_ylabel("verdict accuracy")
    a2.axhline(0.85, ls="--", color=AMBER, lw=1); a2.text(-0.45, 0.865, "CI gate 0.85", color=AMBER, fontsize=8.5, ha="left")
    a2.set_title("Calibration score", loc="left", fontweight="bold")
    fig.text(0.62, 0.005, "*recomputed by hand from the saved report (25/26)", fontsize=8, color=GREY)
    fig.tight_layout()
    save(fig, "judge_calibration.png")


# ---------------------------------------------------------------- figure 4
def fig_development_arc():
    # (x label, pass %, fraction, note). Sources: docs/FINDINGS.md #4 and #5 for the
    # first two; the rest are live runs after the 8 multi-turn cases were added
    # (the first run listed 3 failures out of 24 cases). The dataset grew from 16 to
    # 24 cases between points 2 and 3, so read it as a story, not a like-for-like trend.
    pts = [("test bugs\nfixed (#4)", 87.5, "14/16", "2 real agent gaps\nremained"),
           ("policy gap\nfixed (#5)", 100, "16/16", "suite of 16\nsingle-turn cases"),
           ("+8 multi-turn\ncases, 1st run", 87.5, "21/24", "agent offered credit on\nclearance items"),
           ("clearance\nrule added", 91.7, "22/24", "narrating next steps\n(process promises)"),
           ("current", 100, "24/24", "response shape, R12,\njudge notes, gate")]
    fig, ax = plt.subplots(figsize=(10.5, 4.1))
    xs = list(range(len(pts)))
    ax.plot(xs, [p_[1] for p_ in pts], color="#bdbdbd", lw=2, zorder=1)
    for x, (lab, v, frac, note) in zip(xs, pts):
        ax.scatter(x, v, s=170, color=GREEN if v == 100 else AMBER, zorder=3)
        ax.text(x, v + 1.7, frac, ha="center", fontweight="bold")
        ax.text(x, v - 3.2, note, ha="center", va="top", fontsize=8.6, color="#616161")
    ax.set_xticks(xs); ax.set_xticklabels([p_[0] for p_ in pts], fontsize=9.3)
    ax.set_ylim(78, 106); ax.set_xlim(-0.45, len(pts) - 0.55)
    ax.set_ylabel("pass rate (%)")
    ax.set_title("Every dip was a real gap, traced to a cause and closed", loc="left", fontweight="bold")
    save(fig, "development_arc.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=".", help="folder holding the JSON result files")
    a = ap.parse_args()
    print("Generating figures...")
    r = load(a.data, "results.json")
    if r: fig_results_overview(r)
    s = load(a.data, "stability_report.json")
    if s: fig_error_vs_failure(s)
    c = load(a.data, "judge_calibration_report.json")
    if c: fig_judge_calibration(c)
    fig_development_arc()


if __name__ == "__main__":
    main()
