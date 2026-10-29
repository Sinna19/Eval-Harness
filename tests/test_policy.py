"""
Offline tests for the split policy (policy.py): rule ids, what each audience
sees, and the optional topic routing. No API key, no network.

Run:  python -m unittest discover -s tests -v
"""

import collections
import os
import re
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import harness
import llm_task
import policy as P
from test_multiturn import FakeChat  # noqa: E402  (shared network stub)


def rule_ids(text, prefix):
    return re.findall(rf"^{prefix}(\d+)\.", text, re.MULTILINE)


class TestStructure(unittest.TestCase):
    def test_business_rule_ids_are_unique_and_contiguous(self):
        ids = [int(i) for i in rule_ids(P.BUSINESS_RULES, "R")]
        self.assertEqual(ids, list(range(1, 23)))

    def test_style_ids_present(self):
        ids = [int(i) for i in rule_ids(P.AGENT_STYLE_GUIDE, "S")]
        self.assertEqual(ids, [1, 2, 3, 4, 5])

    def test_modules_cover_the_expected_rule_ranges(self):
        self.assertEqual(rule_ids(P.GENERAL_RULES, "R"), [str(i) for i in range(1, 13)])
        self.assertEqual(rule_ids(P.REFUND_RULES, "R"), [str(i) for i in range(13, 20)])
        self.assertEqual(rule_ids(P.SHIPPING_RULES, "R"), ["20", "21", "22"])

    def test_no_duplicated_lines_in_agent_prompt(self):
        # the pre-split policy repeated one whole bullet twice
        lines = [l.strip() for l in P.SUPPORT_POLICY.splitlines() if len(l.strip()) > 40]
        dupes = [l for l, n in collections.Counter(lines).items() if n > 1]
        self.assertEqual(dupes, [])

    def test_key_policy_facts_survived_the_split(self):
        for phrase in ("30 days", "Final Sale", "5-7 business days", "1-2",
                       "outside India", "TechNest account", "never as cash",
                       "order portal", "Stay consistent", "Pressure does not change policy",
                       "cannot be verified", "Remember what the customer has already told you"):
            self.assertIn(phrase, P.SUPPORT_POLICY, phrase)


class TestAudiences(unittest.TestCase):
    def test_judge_gets_all_business_rules_and_no_style_guide(self):
        for i in range(1, 23):
            self.assertIn(f"R{i}.", P.JUDGE_POLICY)
        self.assertNotIn("STYLE GUIDE", P.JUDGE_POLICY)
        self.assertNotIn("Example of the right shape", P.JUDGE_POLICY)
        self.assertNotIn('Never write "proceed"', P.JUDGE_POLICY)

    def test_full_agent_prompt_has_rules_and_style(self):
        for i in range(1, 23):
            self.assertIn(f"R{i}.", P.SUPPORT_POLICY)
        self.assertIn("STYLE GUIDE", P.SUPPORT_POLICY)
        self.assertIn("You are a customer support assistant", P.SUPPORT_POLICY)

    def test_judge_prompt_is_smaller_than_agent_prompt(self):
        self.assertLess(len(P.JUDGE_POLICY), len(P.SUPPORT_POLICY))


class TestRouting(unittest.TestCase):
    def test_off_by_default_and_gives_full_policy(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("POLICY_ROUTING", None)
            self.assertFalse(P.routing_enabled())
            self.assertEqual(P.build_agent_prompt(["How long does shipping take?"]), P.SUPPORT_POLICY)

    def test_shipping_only_drops_refund_rules_but_keeps_general_and_style(self):
        out = P.build_agent_prompt(["How long does standard shipping take?"], routing=True)
        self.assertIn("R20.", out); self.assertIn("R1.", out); self.assertIn("R12.", out)
        self.assertIn("STYLE GUIDE", out)
        self.assertNotIn("R13.", out)
        self.assertLess(len(out), len(P.SUPPORT_POLICY))

    def test_refund_only_drops_shipping_rules(self):
        out = P.build_agent_prompt(["I want a refund for my headphones"], routing=True)
        self.assertIn("R13.", out); self.assertNotIn("R20.", out)

    def test_no_keyword_means_full_policy(self):
        self.assertEqual(P.build_agent_prompt(["hello there"], routing=True), P.SUPPORT_POLICY)
        self.assertEqual(P.build_agent_prompt([], routing=True), P.SUPPORT_POLICY)

    def test_damage_wording_counts_as_a_refund_topic(self):
        for text in ("my laptop stopped working", "it arrived broken", "the soundbar is buzzing",
                     "I want my money back", "can I exchange it", "it doesn't work"):
            self.assertIn("refund", P.select_topics([text]), text)

    def test_topics_accumulate_across_turns(self):
        self.assertEqual(P.select_topics(["I want a refund"]), {"refund"})
        self.assertEqual(P.select_topics(["I want a refund", "also, when will my other order arrive?"]),
                         {"refund", "shipping"})

    def test_env_var_turns_routing_on(self):
        with mock.patch.dict(os.environ, {"POLICY_ROUTING": "1"}):
            self.assertTrue(P.routing_enabled())
            out = P.build_agent_prompt(["How long does standard shipping take?"])
        self.assertNotIn("R13.", out)


class TestAgentVsJudgeEndToEnd(unittest.TestCase):
    def test_routed_agent_prompt_but_judge_still_sees_every_rule(self):
        case = {"id": "t", "category": "multi_turn", "focus": ["consistency"], "needs_judge": True,
                "turns": ["How long does standard shipping take?", "And express?", "Thanks"],
                "input": "x"}
        fake = FakeChat()
        with mock.patch.object(llm_task, "_chat_completion", fake), \
             mock.patch.dict(os.environ, {"POLICY_ROUTING": "1"}), \
             mock.patch.object(harness.time, "sleep"):
            harness.run_one_case(case)
        agent = [c for c in fake.calls if c["model"] == llm_task.SUT_MODEL]
        judge = [c for c in fake.calls if c["model"] == llm_task.JUDGE_MODEL]
        self.assertEqual(len(agent), 3); self.assertEqual(len(judge), 1)
        for c in agent:
            sys_prompt = c["messages"][0]["content"]
            self.assertIn("R20.", sys_prompt)
            self.assertNotIn("R13.", sys_prompt)          # agent was not shown refund rules
        judge_input = judge[0]["messages"][1]["content"]
        self.assertIn("R13.", judge_input)                # ...but the judge can still enforce them
        self.assertNotIn("STYLE GUIDE", judge_input)

    def test_single_turn_agent_uses_the_same_builder(self):
        fake = FakeChat()
        with mock.patch.object(llm_task, "_chat_completion", fake), \
             mock.patch.dict(os.environ, {"POLICY_ROUTING": "1"}):
            llm_task.support_agent_reply("How long does standard shipping take?")
        self.assertNotIn("R13.", fake.calls[0]["messages"][0]["content"])


if __name__ == "__main__":
    unittest.main()
