"""
This is the "system under test" — the actual AI feature we're evaluating.
It's kept deliberately simple: one function that takes a customer message
and returns the assistant's reply, using the policy as its system prompt.

Uses Groq's API (OpenAI-compatible, running Llama 3.1). Chosen over Gemini
after testing both: Gemini's free tier caps at 20 requests/DAY per model,
which isn't enough to iterate on a growing test suite. Groq's free tier has
a real usable limit for this kind of work.

Swap `call_llm` for any other provider without touching the harness code,
as long as it still returns a plain string.
"""

import os
import time
import requests
from policy import SUPPORT_POLICY

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "openai/gpt-oss-20b"

MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 10


def call_llm(system_prompt: str, user_message: str) -> str:
    """Calls Groq's chat completions endpoint. Returns the reply text.
    Retries with backoff if rate-limited (HTTP 429)."""
    if not GROQ_API_KEY:
        raise RuntimeError(
            "Set the GROQ_API_KEY environment variable before running the harness. "
            "Get a free key at https://console.groq.com/keys"
        )

    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.3,
    }

    backoff = INITIAL_BACKOFF_SECONDS
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=30)

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


def support_agent_reply(customer_message: str) -> str:
    """The function under test: one customer message in, one reply out."""
    return call_llm(SUPPORT_POLICY, customer_message)
