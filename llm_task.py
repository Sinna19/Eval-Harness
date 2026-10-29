"""
This is the "system under test" — the actual AI feature we're evaluating.
It's kept deliberately simple: functions that take a customer message (or a
whole conversation so far) and return the assistant's reply, using the
policy as its system prompt.


Swap `call_llm` for any other provider without touching the harness code,
as long as it still returns a plain string.
"""

import os
import time
import requests
from policy import build_agent_prompt

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Two separate model configs: SUT_MODEL is the system under test, JUDGE_MODEL
# grades its output. These must NOT be the same model -- a model judging its
# own replies is prone to self-preferencing bias (rating its own output more
# favorably, missing its own blind spots). JUDGE_MODEL is deliberately a
# different model family (Qwen, not OpenAI GPT-OSS like the SUT) to reduce
# family-correlated blind spots, though both are still served by Groq; see
# scoring.py / README for the known limitation this leaves. Groq's available
# model list changes over time -- check https://console.groq.com/docs/models
# or GET /openai/v1/models if either of these 404s.
SUT_MODEL = os.environ.get("SUT_MODEL", "openai/gpt-oss-120b")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "qwen/qwen3.8-27b")

MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 10


def _chat_completion(payload: dict) -> str:
    """Posts one chat-completions payload to Groq and returns the reply text.
    Retries with exponential backoff on HTTP 429. Isolated in its own
    function so tests can stub the network layer without touching anything
    above it (call_chat / call_llm / the support agent)."""
    if not GROQ_API_KEY:
        raise RuntimeError(
            "Set the GROQ_API_KEY environment variable before running the harness. "
            "Get a free key at https://console.groq.com/keys"
        )

    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    backoff = INITIAL_BACKOFF_SECONDS
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=60)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_error = e
            if attempt < MAX_RETRIES:
                print(f"    Network error ({type(e).__name__}), waiting {backoff}s before retry...")
                time.sleep(backoff)
                backoff *= 2
                continue
            raise
           
        if resp.status_code == 429:
            last_error = requests.exceptions.HTTPError(
                f"429 rate-limited (attempt {attempt}/{MAX_RETRIES}): {resp.text}"
            )
            if attempt < MAX_RETRIES:
                print(f"    Rate limited, waiting {backoff}s before retry...")
                time.sleep(backoff)
                backoff *= 2
                continue
            raise last_error

        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    raise last_error


def call_chat(system_prompt: str, messages: list, model: str = SUT_MODEL,
              reasoning_effort: str | None = None) -> str:
    """Multi-turn version of call_llm: `messages` is the full conversation so
    far as [{"role": "user"|"assistant", "content": str}, ...], ending with
    a user message. The system prompt is prepended on every call -- the model
    is stateless, so the ONLY thing that gives it memory of earlier turns is
    this list. That is exactly why a multi-turn eval has to replay the whole
    history each time rather than sending the latest message alone.

    reasoning_effort: for reasoning-capable models (e.g. qwen/qwen3.8-27b),
    pass "none" to disable chain-of-thought so only the final answer comes
    back. Without this, a reasoning model's full internal deliberation --
    including draft answers it later revises -- can leak into the
    response content, which breaks any downstream parsing that assumes
    "the text" is just the final answer. Found for real: JUDGE_MODEL
    returned a multi-paragraph <think>-style trace instead of the
    requested one-line REASON/VERDICT format -- see scoring.py."""
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system_prompt}, *messages],
        "temperature": 0.3,
    }
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort
    return _chat_completion(payload)


def call_llm(system_prompt: str, user_message: str, model: str = SUT_MODEL,
             reasoning_effort: str | None = None) -> str:
    """Single-turn convenience wrapper around call_chat. Signature and
    behavior unchanged from before multi-turn support was added."""
    return call_chat(system_prompt, [{"role": "user", "content": user_message}],
                     model=model, reasoning_effort=reasoning_effort)


def support_agent_reply(customer_message: str) -> str:
    return call_llm(build_agent_prompt([customer_message]), customer_message,
                    model=SUT_MODEL)

def support_agent_chat_reply(history: list) -> str:
    customer_texts = [m["content"] for m in history if m["role"] == "user"]
    return call_chat(build_agent_prompt(customer_texts), history, model=SUT_MODEL)
