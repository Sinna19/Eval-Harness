"""Offline tests: llm_task._chat_completion retries transient network errors."""

import os
import sys
import unittest
from unittest import mock

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import llm_task


class Resp:
    status_code = 200
    text = ""

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": "ok"}}]}


class TestNetworkRetry(unittest.TestCase):
    def setUp(self):
        p1 = mock.patch.object(llm_task, "GROQ_API_KEY", "test-key")
        p2 = mock.patch.object(llm_task.time, "sleep")
        self.sleep = p2.start(); p1.start()
        self.addCleanup(p1.stop); self.addCleanup(p2.stop)

    def test_read_timeout_then_success_is_retried(self):
        post = mock.Mock(side_effect=[requests.exceptions.ReadTimeout("t"), Resp()])
        with mock.patch.object(llm_task.requests, "post", post):
            self.assertEqual(llm_task._chat_completion({"model": "m", "messages": []}), "ok")
        self.assertEqual(post.call_count, 2)

    def test_connection_reset_then_success_is_retried(self):
        post = mock.Mock(side_effect=[requests.exceptions.ConnectionError("reset"), Resp()])
        with mock.patch.object(llm_task.requests, "post", post):
            self.assertEqual(llm_task._chat_completion({"model": "m", "messages": []}), "ok")

    def test_persistent_network_failure_raises_after_max_retries(self):
        post = mock.Mock(side_effect=requests.exceptions.ReadTimeout("t"))
        with mock.patch.object(llm_task.requests, "post", post):
            with self.assertRaises(requests.exceptions.ReadTimeout):
                llm_task._chat_completion({"model": "m", "messages": []})
        self.assertEqual(post.call_count, llm_task.MAX_RETRIES)

    def test_backoff_doubles_between_network_retries(self):
        post = mock.Mock(side_effect=[requests.exceptions.ReadTimeout("t")] * 2 + [Resp()])
        with mock.patch.object(llm_task.requests, "post", post):
            llm_task._chat_completion({"model": "m", "messages": []})
        waits = [c.args[0] for c in self.sleep.call_args_list]
        self.assertEqual(waits, [llm_task.INITIAL_BACKOFF_SECONDS, llm_task.INITIAL_BACKOFF_SECONDS * 2])


if __name__ == "__main__":
    unittest.main()
