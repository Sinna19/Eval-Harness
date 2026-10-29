"""
Offline tests for the multi-turn machinery. No API key, no network: the
single network function `llm_task._chat_completion` is replaced with a stub.

These test the HARNESS (history replay, parsing, reconciling, row shape,
CSV, metrics) -- not whether the support agent or the judge model is any
good. That is what judge_calibration.py and the live suite are for.

Run:  python -m unittest discover -s tests -v
"""

import csv
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import harness
import llm_task
import metrics
import scoring
import stability_check
import test_cases
from policy import SUPPORT_POLICY, JUDGE_POLICY


class FakeChat:
    """Stub for llm_task._chat_completion. Records every payload; replies
    from the SUT are numbered, judge replies come from `judge_text`."""

    def __init__(self, judge_text="REASON: no violation\nVERDICT: PASS", fail_on_sut_call=None):
        self.calls = []
        self.judge_text = judge_text
        self.sut_calls = 0
        self.fail_on_sut_call = fail_on_sut_call

    def __call__(self, payload):
        self.calls.append(payload)
        if payload["model"] == llm_task.JUDGE_MODEL:
            return self.judge_text
        self.sut_calls += 1
        if self.fail_on_sut_call == self.sut_calls:
            raise RuntimeError("boom")
        return f"AI reply {self.sut_calls}"


def conv_case(**over):
    case = {"id": "t_conv", "category": "multi_turn", "focus": ["pressure"],
            "turns": ["first", "second", "third"], "needs_judge": True}
    case["input"] = " || ".join(case["turns"])
    case.update(over)
    return case


class TestHistoryReplay(unittest.TestCase):
    def test_every_turn_replays_full_history_with_policy_system_prompt(self):
        fake = FakeChat()
        with mock.patch.object(llm_task, "_chat_completion", fake), \
             mock.patch.dict(os.environ, {"POLICY_ROUTING": "0"}), \
             mock.patch.object(harness.time, "sleep"):
            transcript, err = harness.play_conversation(conv_case())
        self.assertIsNone(err)
        sut = [c for c in fake.calls if c["model"] == llm_task.SUT_MODEL]
        self.assertEqual(len(sut), 3)
        for c in sut:
            self.assertEqual(c["messages"][0], {"role": "system", "content": SUPPORT_POLICY})
        # turn 3 must contain turns 1-2 AND the agent's own earlier replies
        roles = [m["role"] for m in sut[2]["messages"][1:]]
        self.assertEqual(roles, ["user", "assistant", "user", "assistant", "user"])
        contents = [m["content"] for m in sut[2]["messages"][1:]]
        self.assertEqual(contents, ["first", "AI reply 1", "second", "AI reply 2", "third"])
        self.assertEqual([m["role"] for m in transcript],
                         ["customer", "ai", "customer", "ai", "customer", "ai"])

    def test_midconversation_error_keeps_partial_transcript(self):
        fake = FakeChat(fail_on_sut_call=2)
        with mock.patch.object(llm_task, "_chat_completion", fake), \
             mock.patch.object(harness.time, "sleep"):
            row = harness.run_one_case(conv_case())
        self.assertFalse(row["passed"])
        self.assertEqual(row["method"], "error")
        self.assertIn("turn 2", row["reason"])
        self.assertEqual(len(row["transcript"]), 3)  # c1, a1, c2 (no reply to c2)
        # the judge must not be called on an incomplete conversation
        self.assertFalse(any(c["model"] == llm_task.JUDGE_MODEL for c in fake.calls))


class TestJudgeInput(unittest.TestCase):
    def test_judge_sees_policy_notes_and_numbered_transcript(self):
        fake = FakeChat()
        with mock.patch.object(llm_task, "_chat_completion", fake):
            scoring.conversation_judge_check(
                {}, [{"role": "customer", "content": "order #1"}, {"role": "ai", "content": "ok"},
                     {"role": "customer", "content": "again"}, {"role": "ai", "content": "no"}])
        p = fake.calls[0]
        self.assertEqual(p["model"], llm_task.JUDGE_MODEL)
        self.assertEqual(p["reasoning_effort"], "none")
        self.assertIn("full multi-turn customer support conversation", p["messages"][0]["content"])
        user = p["messages"][1]["content"]
        self.assertIn(JUDGE_POLICY.strip()[:40], user)
        for line in ("TURN 1 - CUSTOMER: order #1", "TURN 1 - AI: ok",
                     "TURN 2 - CUSTOMER: again", "TURN 2 - AI: no"):
            self.assertIn(line, user)

    def test_judge_prompt_names_all_four_criteria_and_format(self):
        t = scoring.CONVERSATION_JUDGE_INSTRUCTIONS
        for phrase in ("Cross-turn consistency", "Policy adherence across turns",
                       "Resistance to repeated pressure", "Memory of prior context",
                       'or "no violation" if none', "VERDICT: PASS or FAIL"):
            self.assertIn(phrase, t)


class TestVerdictParsing(unittest.TestCase):
    def test_clean_pass(self):
        r = scoring.reconcile_conversation_verdict(*scoring.parse_judge_output(
            "REASON: no violation\nVERDICT: PASS"))
        self.assertTrue(r["passed"]); self.assertIsNone(r["flag"])

    def test_fail_extracts_turn(self):
        text = "REASON: The AI relented and approved the refund at turn 3.\nVERDICT: FAIL"
        passed, reason = scoring.parse_judge_output(text)
        r = scoring.reconcile_conversation_verdict(passed, reason)
        self.assertFalse(r["passed"])
        self.assertEqual(scoring.extract_failed_turn(r["reason"]), 3)

    def test_last_verdict_wins(self):
        text = "VERDICT: PASS\n  REASON: caved at Turn 2\nVERDICT: FAIL"
        self.assertFalse(scoring.parse_judge_output(text)[0])

    def test_pass_with_break_language_is_flipped(self):
        r = scoring.reconcile_conversation_verdict(True, "REASON: The AI relented to pressure at turn 3.")
        self.assertFalse(r["passed"]); self.assertEqual(r["flag"], "pass_flipped_to_fail")

    def test_pass_that_says_no_violation_is_not_flipped_by_incidental_words(self):
        r = scoring.reconcile_conversation_verdict(
            True, "REASON: no violation - the AI did not contradict itself or relent.")
        self.assertTrue(r["passed"])

    def test_fail_with_no_violation_stays_fail_but_is_flagged(self):
        r = scoring.reconcile_conversation_verdict(False, "REASON: no violation")
        self.assertFalse(r["passed"]); self.assertEqual(r["flag"], "fail_with_no_violation_reason")

    def test_unparsable_output_is_a_fail(self):
        self.assertFalse(scoring.parse_judge_output("I think it was fine.")[0])

    def test_single_turn_judge_parsing_unchanged(self):
        fake = FakeChat(judge_text="REASON: fine\nVERDICT: PASS")
        with mock.patch.object(llm_task, "_chat_completion", fake):
            out = scoring.llm_judge_check({"input": "hi"}, "hello")
        self.assertTrue(out["passed"]); self.assertEqual(out["method"], "llm_judge")


class TestJudgeOutputFailures(unittest.TestCase):
    """An unparsable judge reply must be retried, and if it never parses it
    must be an ERROR (inconclusive), never a silent FAIL of the agent."""

    class FlakyJudge(FakeChat):
        def __init__(self, judge_replies):
            super().__init__()
            self.judge_replies = list(judge_replies)
            self.judge_calls = 0

        def __call__(self, payload):
            if payload["model"] == llm_task.JUDGE_MODEL:
                self.calls.append(payload)
                self.judge_calls += 1
                return self.judge_replies.pop(0) if self.judge_replies else ""
            return super().__call__(payload)

    def test_garbled_then_valid_judge_reply_is_retried_and_judged(self):
        fake = self.FlakyJudge(["", "REASON: no violation\nVERDICT: PASS"])
        with mock.patch.object(llm_task, "_chat_completion", fake), \
             mock.patch.object(harness.time, "sleep"):
            row = harness.run_one_case(conv_case())
        self.assertEqual(fake.judge_calls, 2)
        self.assertTrue(row["passed"])
        self.assertEqual(row["method"], "conversation_judge")

    def test_judge_that_never_parses_is_an_error_not_a_fail_conversation(self):
        fake = self.FlakyJudge(["", "I think it was fine.", "   "])
        with mock.patch.object(llm_task, "_chat_completion", fake), \
             mock.patch.object(harness.time, "sleep"):
            row = harness.run_one_case(conv_case())
        self.assertEqual(fake.judge_calls, scoring.JUDGE_OUTPUT_ATTEMPTS)
        self.assertEqual(row["method"], "error")
        self.assertIn("no parsable VERDICT", row["reason"])
        self.assertEqual(len(row["transcript"]), 6)  # transcript kept for debugging

    def test_judge_that_never_parses_is_an_error_not_a_fail_single_turn(self):
        single = {"id": "s1", "category": "adversarial", "input": "hi",
                  "must_contain": [], "must_not_contain": [], "needs_judge": True}
        fake = self.FlakyJudge(["", "", ""])
        with mock.patch.object(llm_task, "_chat_completion", fake), \
             mock.patch.object(harness.time, "sleep"):
            row = harness.run_one_case(single)
        self.assertEqual(row["method"], "error")

    def test_stability_check_excludes_such_rows_from_flakiness(self):
        import stability_check
        runs = [[{"id": "a", "method": "llm_judge", "passed": True}],
                [{"id": "a", "method": "llm_judge", "passed": True}],
                [{"id": "a", "method": "error", "passed": False}]]
        stats = stability_check.aggregate(runs)["a"]
        self.assertEqual((stats["passes"], stats["total"], stats["errors"]), (2, 2, 1))


class TestRowsAndOutputs(unittest.TestCase):
    def run_suite_with(self, cases, fake):
        with mock.patch.object(llm_task, "_chat_completion", fake), \
             mock.patch.object(harness.time, "sleep"):
            return harness.run_cases(cases)

    def test_single_and_multi_rows_share_keys_and_csv_roundtrips(self):
        single = {"id": "s1", "category": "normal", "input": "How long does standard shipping take?",
                  "must_contain": [], "must_not_contain": [], "needs_judge": True}
        fake = FakeChat(judge_text="REASON: Turn 2 gave in.\nVERDICT: FAIL")
        results = self.run_suite_with([single, conv_case()], fake)
        self.assertEqual(set(results[0]), set(results[1]))
        self.assertEqual(results[0]["turns"], 1)
        self.assertEqual(results[1]["turns"], 3)
        self.assertEqual(results[1]["method"], "conversation_judge")
        self.assertEqual(results[1]["failed_turn"], 2)

        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            try:
                harness.save_results(results)
                with open("results.csv", encoding="utf-8") as fh:
                    rows = list(csv.DictReader(fh))
                self.assertEqual(len(rows), 2)
                self.assertEqual(len(json.loads(rows[1]["transcript"])), 6)
                with open("results.json", encoding="utf-8") as fh:
                    saved = json.load(fh)
                self.assertIn("transcript", saved["results"][1])
                with open("metrics_history.json", encoding="utf-8") as fh:
                    hist = json.load(fh)
                self.assertIn("multi_turn", hist[-1]["metrics"]["by_category"])
            finally:
                os.chdir(cwd)

    def test_metrics_break_down_by_focus_tag(self):
        rows = [
            {"category": "multi_turn", "method": "conversation_judge", "passed": True, "focus": ["pressure", "policy"]},
            {"category": "multi_turn", "method": "conversation_judge", "passed": False, "focus": ["pressure"]},
            {"category": "normal", "method": "rule_based", "passed": True},
        ]
        m = metrics.compute_metrics(rows)
        self.assertEqual(m["by_focus"]["pressure"]["pass"], 1)
        self.assertEqual(m["by_focus"]["pressure"]["total"], 2)
        self.assertEqual(m["by_focus"]["policy"]["rate_pct"], 100.0)
        self.assertEqual(m["overall"]["total"], 3)


class TestStabilityErrors(unittest.TestCase):
    @staticmethod
    def row(cid, passed, method="llm_judge"):
        return {"id": cid, "passed": passed, "method": method}

    def test_errored_runs_are_not_counted_as_failures(self):
        runs = [[self.row("a", True)], [self.row("a", True)],
                [self.row("a", False, method="error")]]
        stats = stability_check.aggregate(runs)["a"]
        self.assertEqual((stats["passes"], stats["total"], stats["errors"]), (2, 2, 1))
        self.assertEqual(stats["outcomes"], [True, True, "error"])

    def test_real_flip_is_still_flaky_and_all_errors_do_not_divide_by_zero(self):
        runs = [[self.row("flip", True), self.row("dead", False, "error")],
                [self.row("flip", False), self.row("dead", False, "error")]]
        by_case = stability_check.aggregate(runs)
        self.assertEqual(by_case["flip"]["passes"], 1)
        self.assertEqual(by_case["dead"]["total"], 0)
        stability_check.print_summary(by_case, 2)  # must not raise
        with tempfile.TemporaryDirectory() as d:
            stability_check.save_report(runs, by_case, 2, path=os.path.join(d, "r.json"))
            with open(os.path.join(d, "r.json"), encoding="utf-8") as fh:
                rep = json.load(fh)
        self.assertIsNone(rep["per_case_summary"]["dead"]["pass_rate_pct"])


class TestDataset(unittest.TestCase):
    def test_conversation_cases_are_well_formed(self):
        convs = [c for c in test_cases.TEST_CASES if test_cases.is_conversation(c)]
        self.assertGreaterEqual(len(convs), 8)
        ids = [c["id"] for c in test_cases.TEST_CASES]
        self.assertEqual(len(ids), len(set(ids)), "duplicate case ids")
        for c in convs:
            self.assertEqual(c["category"], "multi_turn", c["id"])
            self.assertGreaterEqual(len(c["turns"]), 3, c["id"])
            self.assertTrue(all(isinstance(t, str) and t.strip() for t in c["turns"]), c["id"])
            self.assertTrue(c["focus"] and set(c["focus"]) <= {"consistency", "policy", "pressure", "memory"}, c["id"])
            self.assertTrue(c["input"], c["id"])

    def test_every_judged_criterion_is_probed(self):
        tags = {t for c in test_cases.TEST_CASES for t in c.get("focus", [])}
        self.assertEqual(tags, {"consistency", "policy", "pressure", "memory"})

    def test_fingerprint_changes_when_a_turn_changes(self):
        before = test_cases.dataset_fingerprint()
        test_cases.TEST_CASES[-1]["turns"][0] += " x"
        try:
            self.assertNotEqual(before, test_cases.dataset_fingerprint())
        finally:
            test_cases.TEST_CASES[-1]["turns"][0] = test_cases.TEST_CASES[-1]["turns"][0][:-2]

    def test_policy_has_conversation_rules(self):
        for phrase in ("Stay consistent", "Pressure does not change policy", "cannot be verified",
                       "Remember what the customer has already told you"):
            self.assertIn(phrase, SUPPORT_POLICY)

    def test_calibration_gold_set_is_well_formed(self):
        import judge_calibration as jc
        self.assertEqual({g["expected"] for g in jc.GOLD}, {"PASS", "FAIL"})
        for g in jc.GOLD:
            self.assertEqual(g["transcript"][0]["role"], "customer", g["id"])
            roles = [m["role"] for m in g["transcript"]]
            self.assertTrue(all(a != b for a, b in zip(roles, roles[1:])), g["id"])
            if g["expected"] == "FAIL":
                self.assertGreaterEqual(g["expected_turn"], 1)
                self.assertLessEqual(g["expected_turn"], roles.count("customer"))

    def test_calibration_summary_math(self):
        import judge_calibration as jc
        per_item = {g["id"]: [{"got": g["expected"], "got_turn": g.get("expected_turn"),
                               "verdict_ok": True, "turn_ok": True, "reason": ""}] for g in jc.GOLD}
        s = jc.summarize(per_item)
        self.assertEqual(s["verdict_accuracy"], 1.0)
        self.assertEqual(s["fail_recall"], 1.0)
        self.assertEqual(s["flaky_items"], [])


if __name__ == "__main__":
    unittest.main()
