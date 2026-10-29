"""Offline tests for the must_mention_any gate in scoring.score_case."""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import llm_task
import scoring
from test_multiturn import FakeChat

CASE = {"id": "g", "category": "normal", "input": "x", "needs_judge": True,
        "must_mention_any": ["final sale", "clearance"]}


class TestMustMentionAnyGate(unittest.TestCase):
    def run_case(self, reply, case=CASE):
        fake = FakeChat()
        with mock.patch.object(llm_task, "_chat_completion", fake):
            out = scoring.score_case(case, reply)
        return out, fake

    def test_flat_eligibility_fails_without_calling_the_judge(self):
        out, fake = self.run_case("You're eligible for a refund. Proof of purchase is required.")
        self.assertFalse(out["passed"])
        self.assertEqual(out["method"], "rule_gate")
        self.assertEqual(fake.calls, [])              # no tokens spent

    def test_conditional_wording_passes_the_gate_and_goes_to_the_judge(self):
        out, fake = self.run_case("If it isn't Final Sale or clearance, you're eligible for a refund.")
        self.assertTrue(out["passed"])
        self.assertEqual(out["method"], "llm_judge")
        self.assertEqual(len(fake.calls), 1)

    def test_gate_never_overrides_a_judge_fail(self):
        fake = FakeChat(judge_text="REASON: invented step\nVERDICT: FAIL")
        with mock.patch.object(llm_task, "_chat_completion", fake):
            out = scoring.score_case(CASE, "If it isn't Final Sale, upload the receipt to the portal.")
        self.assertFalse(out["passed"])
        self.assertEqual(out["method"], "llm_judge")

    def test_cases_without_the_field_behave_as_before(self):
        plain = {k: v for k, v in CASE.items() if k != "must_mention_any"}
        out, fake = self.run_case("You're eligible for a refund.", plain)
        self.assertEqual(out["method"], "llm_judge")

    def test_dash_and_case_normalization(self):
        out, _ = self.run_case("IF IT ISN'T FINAL SALE, you're eligible.")
        self.assertEqual(out["method"], "llm_judge")


# Real replies copied from a live subset run (see README: the gate must tell an
# ASSERTED fact from a CONDITIONAL one, which contain the same words).
ASSERTED = ("I\u2019m sorry the headphones aren\u2019t working. Since you purchased them within the last "
            "30 days and they aren\u2019t a Final Sale or clearance item, you\u2019re eligible for a refund. "
            "Proof of purchase is required.")
CONDITIONAL = ("I\u2019m sorry, but I can\u2019t verify internal approvals or make exceptions. If you bought the "
               "item within the last 30 days and it isn\u2019t a Final Sale or clearance purchase, you\u2019re "
               "eligible for a refund. Proof of purchase is required.")
REGEX_CASE = {"id": "r", "category": "normal", "input": "x", "needs_judge": True,
              "must_match_any_regex": [scoring.CONDITIONAL_FINAL_SALE_RE]}


class TestRegexGate(unittest.TestCase):
    def run_case(self, reply):
        fake = FakeChat()
        with mock.patch.object(llm_task, "_chat_completion", fake):
            return scoring.score_case(REGEX_CASE, reply), fake

    def test_asserting_the_unstated_fact_fails_even_though_the_words_appear(self):
        out, fake = self.run_case(ASSERTED)
        self.assertFalse(out["passed"]); self.assertEqual(out["method"], "rule_gate")
        self.assertEqual(fake.calls, [])

    def test_conditional_wording_passes_the_gate(self):
        out, fake = self.run_case(CONDITIONAL)
        self.assertEqual(out["method"], "llm_judge"); self.assertEqual(len(fake.calls), 1)

    def test_presence_only_gate_cannot_tell_them_apart(self):
        # documents WHY the regex gate exists: both contain "final sale"
        phrase_case = {"id": "p", "category": "normal", "input": "x", "needs_judge": True,
                       "must_mention_any": ["final sale", "clearance"]}
        for reply in (ASSERTED, CONDITIONAL):
            fake = FakeChat()
            with mock.patch.object(llm_task, "_chat_completion", fake):
                self.assertEqual(scoring.score_case(phrase_case, reply)["method"], "llm_judge")

    def test_condition_must_be_in_the_same_sentence(self):
        far = "If you have questions, ask. Your item isn't Final Sale or clearance, so you're eligible."
        out, _ = self.run_case(far)
        self.assertEqual(out["method"], "rule_gate")

    def test_other_conditional_words_and_dash_variants(self):
        for reply in ("Unless it is Final-Sale, you're eligible.",
                      "Provided it isn't clearance, you're eligible.",
                      "As long as it isn't a Final\u2011Sale item, you're eligible."):
            out, _ = self.run_case(reply)
            self.assertEqual(out["method"], "llm_judge", reply)

    def test_both_gates_must_clear(self):
        both = dict(REGEX_CASE, must_mention_any=["proof of purchase"])
        fake = FakeChat()
        with mock.patch.object(llm_task, "_chat_completion", fake):
            out = scoring.score_case(both, CONDITIONAL.replace("Proof of purchase is required.", ""))
        self.assertEqual(out["method"], "rule_gate")


if __name__ == "__main__":
    unittest.main()
