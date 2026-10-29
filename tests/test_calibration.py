"""Offline tests for judge_calibration.summarize: errors are inconclusive, not misses."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import judge_calibration as jc


def ok(g):
    return {"got": g["expected"], "got_turn": g.get("expected_turn"),
            "verdict_ok": True, "turn_ok": True, "reason": ""}


def err():
    return {"got": "ERROR", "got_turn": None, "verdict_ok": False, "turn_ok": False,
            "reason": "429 rate-limited"}


class TestSummarizeErrors(unittest.TestCase):
    def test_errors_do_not_count_as_wrong_verdicts(self):
        per_item = {g["id"]: [ok(g), ok(g)] for g in jc.GOLD}
        first = jc.GOLD[0]["id"]
        per_item[first] = [ok(jc.GOLD[0]), err()]
        s = jc.summarize(per_item)
        self.assertEqual(s["verdict_accuracy"], 1.0)
        self.assertEqual(s["errors"], 1)
        self.assertEqual(s["usable_attempts"], s["attempts"] - 1)
        self.assertEqual(s["items_with_errors"], [first])
        self.assertEqual(s["flaky_items"], [])         # error + ok is not "flaky"

    def test_real_misses_still_count_and_real_flakiness_is_reported(self):
        per_item = {g["id"]: [ok(g), ok(g)] for g in jc.GOLD}
        bad = jc.GOLD[1]
        wrong = {"got": "PASS" if bad["expected"] == "FAIL" else "FAIL", "got_turn": None,
                 "verdict_ok": False, "turn_ok": False, "reason": "x"}
        per_item[bad["id"]] = [ok(bad), wrong]
        s = jc.summarize(per_item)
        self.assertLess(s["verdict_accuracy"], 1.0)
        self.assertEqual(s["flaky_items"], [bad["id"]])
        self.assertEqual(s["errors"], 0)

    def test_all_errors_does_not_crash(self):
        per_item = {g["id"]: [err()] for g in jc.GOLD}
        s = jc.summarize(per_item)
        self.assertEqual(s["usable_attempts"], 0)
        self.assertEqual(s["errors"], len(jc.GOLD))

    def test_default_delay_respects_the_judge_token_budget(self):
        import inspect
        self.assertGreaterEqual(inspect.signature(jc.run_calibration).parameters["delay"].default, 20)


if __name__ == "__main__":
    unittest.main()
