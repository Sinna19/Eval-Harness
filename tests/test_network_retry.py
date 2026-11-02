"""Offline tests for the LiteLLM model layer in llm_task: retries on
transient errors, fail-fast on real errors, credential checks, model-name
resolution, and per-case usage metering. litellm.completion is stubbed, so
no key or network is needed."""

import os
import sys
import unittest
from unittest import mock

import litellm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import harness
import llm_task
import metrics


def fake_response(text="ok", prompt_tokens=100, completion_tokens=20):
    # Real LiteLLM responses carry the provider in _hidden_params; cost
    # calculation relies on it, so the fake does too.
    response = litellm.ModelResponse(
        model="openai/gpt-oss-120b",
        choices=[{"index": 0, "finish_reason": "stop",
                  "message": {"role": "assistant", "content": text}}],
        usage={"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
               "total_tokens": prompt_tokens + completion_tokens},
    )
    response._hidden_params = {"custom_llm_provider": "groq"}
    return response


def err(kind):
    args = dict(message="boom", llm_provider="groq", model="groq/openai/gpt-oss-120b")
    return getattr(litellm.exceptions, kind)(**args)


PAYLOAD = {"model": "groq/openai/gpt-oss-120b", "messages": []}


class TestRetry(unittest.TestCase):
    def setUp(self):
        patches = [
            mock.patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}),
            mock.patch.object(llm_task.time, "sleep"),
        ]
        started = [p.start() for p in patches]
        self.sleep = started[1]
        for p in patches:
            self.addCleanup(p.stop)

    def _run(self, side_effect):
        completion = mock.Mock(side_effect=side_effect)
        with mock.patch.object(llm_task.litellm, "completion", completion):
            return llm_task._chat_completion(dict(PAYLOAD)), completion

    def test_timeout_then_success_is_retried(self):
        reply, completion = self._run([err("Timeout"), fake_response()])
        self.assertEqual(reply, "ok")
        self.assertEqual(completion.call_count, 2)

    def test_connection_error_then_success_is_retried(self):
        reply, _ = self._run([err("APIConnectionError"), fake_response()])
        self.assertEqual(reply, "ok")

    def test_rate_limit_then_success_is_retried(self):
        reply, _ = self._run([err("RateLimitError"), fake_response()])
        self.assertEqual(reply, "ok")

    def test_persistent_failure_raises_after_max_retries(self):
        completion = mock.Mock(side_effect=err("Timeout"))
        with mock.patch.object(llm_task.litellm, "completion", completion):
            with self.assertRaises(litellm.exceptions.Timeout):
                llm_task._chat_completion(dict(PAYLOAD))
        self.assertEqual(completion.call_count, llm_task.MAX_RETRIES)

    def test_non_transient_error_is_not_retried(self):
        completion = mock.Mock(side_effect=err("AuthenticationError"))
        with mock.patch.object(llm_task.litellm, "completion", completion):
            with self.assertRaises(litellm.exceptions.AuthenticationError):
                llm_task._chat_completion(dict(PAYLOAD))
        self.assertEqual(completion.call_count, 1)

    def test_backoff_doubles_between_retries(self):
        self._run([err("Timeout"), err("RateLimitError"), fake_response()])
        waits = [c.args[0] for c in self.sleep.call_args_list]
        self.assertEqual(waits, [llm_task.INITIAL_BACKOFF_SECONDS,
                                 llm_task.INITIAL_BACKOFF_SECONDS * 2])

    def test_litellm_internal_retries_are_disabled(self):
        _, completion = self._run([fake_response()])
        self.assertEqual(completion.call_args.kwargs["num_retries"], 0)


class TestCredentials(unittest.TestCase):
    def test_missing_groq_key_gives_actionable_error(self):
        env = {k: v for k, v in os.environ.items() if k != "GROQ_API_KEY"}
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(RuntimeError, "GROQ_API_KEY"):
                llm_task._chat_completion(dict(PAYLOAD))


class TestResolveModel(unittest.TestCase):
    def test_full_litellm_string_is_unchanged(self):
        self.assertEqual(llm_task.resolve_model("groq/openai/gpt-oss-120b"),
                         "groq/openai/gpt-oss-120b")
        self.assertEqual(llm_task.resolve_model("gemini/gemini-2.5-flash"),
                         "gemini/gemini-2.5-flash")

    def test_legacy_groq_id_is_routed_to_groq_not_openai(self):
        # Pre-LiteLLM SUT_MODEL values were bare Groq ids; "openai/..." must
        # not silently start hitting OpenAI's API.
        self.assertEqual(llm_task.resolve_model("openai/gpt-oss-120b"),
                         "groq/openai/gpt-oss-120b")
        self.assertEqual(llm_task.resolve_model("qwen/qwen3.8-27b"),
                         "groq/qwen/qwen3.8-27b")

    def test_bare_name_defaults_to_groq(self):
        self.assertEqual(llm_task.resolve_model("llama-3.3-70b-versatile"),
                         "groq/llama-3.3-70b-versatile")

    def test_unknown_provider_string_passes_through(self):
        self.assertEqual(llm_task.resolve_model("ollama/llama3.2"), "ollama/llama3.2")


class TestUsageMetering(unittest.TestCase):
    def setUp(self):
        p1 = mock.patch.dict(os.environ, {"GROQ_API_KEY": "test-key"})
        p1.start(); self.addCleanup(p1.stop)

    def test_tokens_cost_and_role_are_recorded(self):
        completion = mock.Mock(return_value=fake_response(prompt_tokens=1000, completion_tokens=200))
        with mock.patch.object(llm_task.litellm, "completion", completion):
            with llm_task.track_usage() as usage:
                llm_task.call_llm("sys", "hi", model=llm_task.SUT_MODEL)
                llm_task.call_llm("sys", "grade", model=llm_task.JUDGE_MODEL)
        self.assertEqual(usage["calls"], 2)
        self.assertEqual(usage["prompt_tokens"], 2000)
        self.assertEqual(usage["completion_tokens"], 400)
        self.assertGreater(usage["cost_usd"], 0)          # both default models are priced
        self.assertEqual(usage["by_role"]["sut"]["calls"], 1)
        self.assertEqual(usage["by_role"]["judge"]["calls"], 1)
        self.assertAlmostEqual(usage["cost_usd"],
                               usage["by_role"]["sut"]["cost_usd"]
                               + usage["by_role"]["judge"]["cost_usd"])

    def test_litellm_built_in_response_cost_is_preferred(self):
        # The cost LiteLLM computed during the call is the source of truth;
        # the harness must not recompute it when it's there.
        response = fake_response()
        response._hidden_params = {"response_cost": 0.123}
        completion = mock.Mock(return_value=response)
        with mock.patch.object(llm_task.litellm, "completion", completion), \
             mock.patch.object(llm_task.litellm, "completion_cost") as recompute:
            with llm_task.track_usage() as usage:
                llm_task.call_llm("sys", "hi", model="groq/openai/gpt-oss-120b")
        self.assertAlmostEqual(usage["cost_usd"], 0.123)
        self.assertEqual(usage["unpriced_calls"], 0)
        recompute.assert_not_called()

    def test_unpriced_model_stays_unpriced_even_with_built_in_cost(self):
        # LiteLLM may report response_cost=0.0 for a model it has no price
        # for; that must still be flagged, not shown as a real $0.
        response = fake_response()
        response._hidden_params = {"response_cost": 0.0}
        completion = mock.Mock(return_value=response)
        with mock.patch.object(llm_task.litellm, "completion", completion):
            with llm_task.track_usage() as usage:
                llm_task.call_llm("sys", "hi", model="ollama/llama3.2")
        self.assertEqual(usage["unpriced_calls"], 1)

    def test_groq_nested_openai_model_name_is_still_priced(self):
        # Regression: a real Groq response says model="openai/gpt-oss-120b"
        # and carries no provider hint we can rely on; cost lookup must not
        # wander off to OpenAI's price table and come back unpriced.
        response = fake_response()
        response._hidden_params = {}
        completion = mock.Mock(return_value=response)
        with mock.patch.object(llm_task.litellm, "completion", completion):
            with llm_task.track_usage() as usage:
                llm_task.call_llm("sys", "hi", model="groq/openai/gpt-oss-120b")
        self.assertEqual(usage["unpriced_calls"], 0)
        self.assertGreater(usage["cost_usd"], 0)

    def test_unpriced_model_counts_as_zero_not_a_crash(self):
        completion = mock.Mock(return_value=fake_response())
        with mock.patch.object(llm_task.litellm, "completion", completion), \
             mock.patch.object(llm_task.litellm, "completion_cost", side_effect=Exception("no price")):
            with llm_task.track_usage() as usage:
                llm_task.call_llm("sys", "hi")
        self.assertEqual(usage["cost_usd"], 0)
        self.assertEqual(usage["unpriced_calls"], 1)

    def test_model_without_pricing_data_is_flagged_unpriced(self):
        completion = mock.Mock(return_value=fake_response())
        with mock.patch.object(llm_task.litellm, "completion", completion):
            with llm_task.track_usage() as usage:
                llm_task.call_llm("sys", "hi", model="ollama/llama3.2")
        self.assertEqual(usage["unpriced_calls"], 1)
        self.assertEqual(usage["by_role"]["other"]["calls"], 1)

    def test_no_tracker_means_no_recording(self):
        completion = mock.Mock(return_value=fake_response())
        with mock.patch.object(llm_task.litellm, "completion", completion):
            self.assertEqual(llm_task.call_llm("sys", "hi"), "ok")   # must not raise

    def test_every_harness_row_carries_usage_columns(self):
        case = {"id": "t1", "category": "normal", "input": "hello",
                "check": lambda reply: True}
        with mock.patch.object(llm_task, "_chat_completion", return_value="fine"), \
             mock.patch.object(harness, "score_case",
                               return_value={"passed": True, "method": "rule_based", "reason": "ok"}):
            row = harness.run_one_case(case)
        for key in ("cost_usd", "sut_cost_usd", "judge_cost_usd", "llm_calls",
                    "prompt_tokens", "completion_tokens", "sut_latency_s", "unpriced_calls"):
            self.assertIn(key, row)

    def test_usage_summary_ignores_pre_litellm_rows_and_error_latency(self):
        rows = [
            {"category": "a", "passed": True, "method": "rule_based"},          # old row
            {"category": "a", "passed": True, "method": "rule_based", "cost_usd": 0.01,
             "sut_cost_usd": 0.008, "judge_cost_usd": 0.002, "llm_calls": 2,
             "prompt_tokens": 10, "completion_tokens": 5, "unpriced_calls": 0, "sut_latency_s": 1.0},
            {"category": "a", "passed": False, "method": "error", "cost_usd": 0.002,
             "sut_cost_usd": 0.002, "judge_cost_usd": 0, "llm_calls": 1,
             "prompt_tokens": 3, "completion_tokens": 0, "unpriced_calls": 0, "sut_latency_s": 30.0},
        ]
        u = metrics.compute_metrics(rows)["usage"]
        self.assertAlmostEqual(u["cost_usd"], 0.012)
        self.assertEqual(u["errors"], 1)
        self.assertEqual(u["sut_latency_p95_s"], 1.0)     # error row's 30s excluded

    def test_results_metadata_records_models(self):
        meta = harness.run_metadata()
        self.assertEqual(meta["sut_model"], llm_task.SUT_MODEL)
        self.assertEqual(meta["judge_model"], llm_task.JUDGE_MODEL)
        self.assertIn("litellm_version", meta)


if __name__ == "__main__":
    unittest.main()
