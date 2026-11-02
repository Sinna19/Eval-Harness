"""
This is the "system under test" — the actual AI feature we're evaluating.
It's kept deliberately simple: functions that take a customer message (or a
whole conversation so far) and return the assistant's reply, using the
policy as its system prompt.

All model calls go through LiteLLM (https://github.com/BerriAI/litellm), so
the agent and the judge can each be ANY provider LiteLLM supports, chosen
by model string alone -- no harness code changes:

    SUT_MODEL=groq/openai/gpt-oss-120b      (default, needs GROQ_API_KEY)
    SUT_MODEL=gemini/gemini-2.5-flash       (needs GEMINI_API_KEY)
    SUT_MODEL=ollama/llama3.2               (local, needs OLLAMA_API_BASE)

Every call is also metered: tokens, list-price cost (LiteLLM's pricing map)
and latency are collected per case via `track_usage()`, so a results file
says not just *whether* a model held the policy but what it cost and how
fast it was.
"""

import os
import time
from contextlib import contextmanager

import litellm

from policy import build_agent_prompt

# Drop params a given provider doesn't support (e.g. reasoning_effort on a
# non-reasoning model) instead of failing the call -- the harness passes the
# same payload shape to every provider.
litellm.drop_params = True
litellm.suppress_debug_info = True

DEFAULT_PROVIDER = "groq"


def resolve_model(name: str) -> str:
    """Turns a model name into a LiteLLM model string.

    Backward compatible with the pre-LiteLLM env values, which were bare
    Groq model ids. Those are ambiguous under LiteLLM: "openai/gpt-oss-120b"
    would be routed to OpenAI, not Groq. So a name is treated as a Groq id
    when LiteLLM knows "groq/<name>" but not "<name>" itself, or when it has
    no provider prefix at all."""
    name = name.strip()
    if name in litellm.model_cost:
        return name
    if f"{DEFAULT_PROVIDER}/{name}" in litellm.model_cost or "/" not in name:
        return f"{DEFAULT_PROVIDER}/{name}"
    return name


# Two separate model configs: SUT_MODEL is the system under test, JUDGE_MODEL
# grades its output. These must NOT be the same model -- a model judging its
# own replies is prone to self-preferencing bias (rating its own output more
# favorably, missing its own blind spots). JUDGE_MODEL is deliberately a
# different model family (Qwen, not OpenAI GPT-OSS like the SUT) to reduce
# family-correlated blind spots. Both default to Groq; with LiteLLM the judge
# can also come from a different *provider* (e.g. JUDGE_MODEL=gemini/...),
# which removes the shared-serving limitation noted in the README.
SUT_MODEL = resolve_model(os.environ.get("SUT_MODEL", "groq/openai/gpt-oss-120b"))
JUDGE_MODEL = resolve_model(os.environ.get("JUDGE_MODEL", "groq/qwen/qwen3.8-27b"))

MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 10
REQUEST_TIMEOUT_SECONDS = 60

# Transient failures worth backing off and retrying. Anything else (bad key,
# unknown model, malformed request) fails immediately.
RETRYABLE_ERRORS = (
    litellm.exceptions.RateLimitError,
    litellm.exceptions.Timeout,
    litellm.exceptions.APIConnectionError,
    litellm.exceptions.ServiceUnavailableError,
)


# ---------------------------------------------------------------- usage meter

_usage_stack: list[dict] = []


def _empty_usage() -> dict:
    return {
        "calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
        "cost_usd": 0.0, "unpriced_calls": 0,
        "by_role": {role: {"calls": 0, "cost_usd": 0.0, "latency_s": 0.0}
                    for role in ("sut", "judge", "other")},
    }


@contextmanager
def track_usage():
    """Collects tokens, cost and latency for every model call made inside
    the block. Nestable: an outer tracker sees everything an inner one does."""
    record = _empty_usage()
    _usage_stack.append(record)
    try:
        yield record
    finally:
        _usage_stack.remove(record)


def _role_of(model: str) -> str:
    if model == SUT_MODEL:
        return "sut"
    if model == JUDGE_MODEL:
        return "judge"
    return "other"


def _response_cost(model: str, response) -> tuple[float, bool]:
    """Returns (cost_usd, priced) for one completion.

    Models missing from LiteLLM's pricing map (e.g. a local Ollama model)
    are counted as unpriced rather than silently reported as free.

    For priced models, the primary source is the cost LiteLLM computes
    itself during the call (response._hidden_params["response_cost"]): it
    is the number LiteLLM's own spend tracking uses, and it handles cached
    and reasoning tokens. Recomputing it with completion_cost() is only a
    fallback, and needs the provider passed explicitly: Groq responses
    report nested model names like "openai/gpt-oss-120b" with no provider
    hint, so completion_cost() would infer provider "openai" from that
    prefix and fail. Found for real on the first live run, then confirmed
    against a live Groq response: response_cost was set, provider hint
    was None, bare completion_cost() raised."""
    if model not in litellm.model_cost:
        return 0.0, False

    built_in = (getattr(response, "_hidden_params", None) or {}).get("response_cost")
    if isinstance(built_in, (int, float)):
        return float(built_in), True

    try:
        provider = litellm.get_llm_provider(model)[1]
        return float(litellm.completion_cost(completion_response=response, model=model,
                                             custom_llm_provider=provider)), True
    except Exception:
        return 0.0, False


def _record_usage(model: str, response, latency_s: float) -> None:
    if not _usage_stack:
        return
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    cost, priced = _response_cost(model, response)

    role = _role_of(model)
    for record in _usage_stack:
        record["calls"] += 1
        record["prompt_tokens"] += prompt_tokens
        record["completion_tokens"] += completion_tokens
        record["cost_usd"] += cost
        record["unpriced_calls"] += 0 if priced else 1
        bucket = record["by_role"][role]
        bucket["calls"] += 1
        bucket["cost_usd"] += cost
        bucket["latency_s"] += latency_s


# --------------------------------------------------------------- model calls

def _check_credentials(model: str) -> None:
    env = litellm.validate_environment(model)
    missing = env.get("missing_keys") or []
    # OLLAMA_API_BASE has a sensible default (localhost:11434); only treat
    # real credentials as hard requirements.
    missing = [k for k in missing if not k.endswith("_API_BASE")]
    if missing:
        hint = " Get a free key at https://console.groq.com/keys" if "GROQ_API_KEY" in missing else ""
        raise RuntimeError(
            f"Set {', '.join(missing)} before running the harness with model '{model}'.{hint}"
        )


def _chat_completion(payload: dict) -> str:
    """Sends one chat-completions payload through LiteLLM and returns the
    reply text. Retries transient errors (rate limits, timeouts, dropped
    connections, 503s) with exponential backoff. Isolated in its own
    function so tests can stub the network layer without touching anything
    above it (call_chat / call_llm / the support agent)."""
    model = payload["model"]
    _check_credentials(model)

    backoff = INITIAL_BACKOFF_SECONDS
    for attempt in range(1, MAX_RETRIES + 1):
        started = time.perf_counter()
        try:
            # num_retries=0: retrying is done here, so backoff timing stays
            # visible, testable and identical across providers.
            response = litellm.completion(**payload, num_retries=0,
                                          timeout=REQUEST_TIMEOUT_SECONDS)
        except RETRYABLE_ERRORS as e:
            if attempt == MAX_RETRIES:
                raise
            kind = "Rate limited" if isinstance(e, litellm.exceptions.RateLimitError) \
                else f"Transient error ({type(e).__name__})"
            print(f"    {kind}, waiting {backoff}s before retry {attempt + 1}/{MAX_RETRIES}...")
            time.sleep(backoff)
            backoff *= 2
            continue

        _record_usage(model, response, time.perf_counter() - started)
        return response.choices[0].message.content or ""


def call_chat(system_prompt: str, messages: list, model: str | None = None,
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
    requested one-line REASON/VERDICT format -- see scoring.py.
    Providers that don't support the parameter simply drop it."""
    payload = {
        "model": resolve_model(model or SUT_MODEL),
        "messages": [{"role": "system", "content": system_prompt}, *messages],
        "temperature": 0.3,
    }
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort
    return _chat_completion(payload)


def call_llm(system_prompt: str, user_message: str, model: str | None = None,
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
