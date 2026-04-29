# Copyright 2026 Aevyra AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LLM backends for Origin.

Origin's public API takes any callable ``(prompt: str) -> str`` as its LLM.
This module provides convenience factories for the common cases:

    - ``anthropic_llm`` — Claude via the Anthropic SDK (default recommendation)
    - ``openai_llm``    — any OpenAI-compatible endpoint (OpenAI, OpenRouter, Together, vLLM, …)

Both factories return class instances that satisfy the ``LLMFn`` protocol and
additionally expose a ``tokens_used`` attribute for token accounting::

    llm = anthropic_llm()
    text = llm("What went wrong?")
    print(llm.tokens_used)   # total tokens consumed so far

Attribution is a reasoning task that benefits from determinism, so all
factories default to ``temperature=0.0``. Advanced users can override.

Interop with Reflex::

    from aevyra_reflex import LLM
    reflex_llm = LLM(model="claude-sonnet-4-5")
    origin = Origin(llm=lambda p: reflex_llm.generate(p, temperature=0.0))

    # Note: a plain lambda won't have tokens_used — token accounting only
    # works with the factories in this module (or any callable that exposes
    # a ``tokens_used: int`` attribute).
"""

from __future__ import annotations

from typing import Callable

LLMFn = Callable[[str], str]
"""The minimal LLM interface Origin depends on: a callable mapping prompt → text.

Callables returned by :func:`anthropic_llm` and :func:`openai_llm` additionally
expose a ``tokens_used: int`` attribute that accumulates across calls. Plain
lambdas or closures work as ``LLMFn`` but won't contribute to token accounting.
"""


class _AnthropicLLM:
    """Callable LLM backed by the Anthropic SDK with built-in token tracking."""

    def __init__(self, client: object, model: str, max_tokens: int, temperature: float) -> None:
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self.tokens_used: int = 0

    def __call__(self, prompt: str) -> str:
        resp = self._client.messages.create(  # type: ignore[attr-defined]
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        usage = getattr(resp, "usage", None)
        if usage is not None:
            self.tokens_used += getattr(usage, "input_tokens", 0) + getattr(
                usage, "output_tokens", 0
            )
        return resp.content[0].text


class _OpenAILLM:
    """Callable LLM backed by any OpenAI-compatible endpoint with built-in token tracking."""

    def __init__(self, client: object, model: str, max_tokens: int, temperature: float) -> None:
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self.tokens_used: int = 0

    def __call__(self, prompt: str) -> str:
        resp = self._client.chat.completions.create(  # type: ignore[attr-defined]
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        usage = getattr(resp, "usage", None)
        if usage is not None:
            self.tokens_used += getattr(usage, "prompt_tokens", 0) + getattr(
                usage, "completion_tokens", 0
            )
        msg = resp.choices[0].message
        content = msg.content or ""
        if not content:
            # Some reasoning models (e.g. Qwen3-thinking on OpenRouter) return
            # their answer in reasoning_content when content is empty.
            content = getattr(msg, "reasoning_content", None) or ""
        if not content:
            # Still empty — print diagnostic so the user can investigate.
            import sys as _sys
            finish_reason = getattr(resp.choices[0], "finish_reason", None)
            _sys.stderr.write(
                f"Warning: model '{self._model}' returned empty content "
                f"(finish_reason={finish_reason!r}). "
                f"Full message fields: {list(vars(msg).keys())}\n"
            )
        return content


def anthropic_llm(
    model: str = "claude-sonnet-4-5",
    *,
    api_key: str | None = None,
    max_tokens: int = 16384,
    temperature: float = 0.0,
) -> _AnthropicLLM:
    """Factory: Claude via the Anthropic Python SDK.

    Requires the ``anthropic`` extra::

        pip install aevyra-origin[anthropic]

    Args:
        model:       Anthropic model ID. Defaults to ``"claude-sonnet-4-5"``.
        api_key:     API key. If None, the Anthropic SDK picks up
                     ``ANTHROPIC_API_KEY`` from the environment.
        max_tokens:  Max output tokens per call.
        temperature: Sampling temperature. Defaults to 0.0 (deterministic)
                     because attribution is a reasoning task and stability
                     across reruns matters more than variety.

    Returns:
        A callable ``(prompt: str) -> str`` with a ``tokens_used`` attribute.
        Satisfies :data:`LLMFn`.
    """
    try:
        from anthropic import Anthropic
    except ImportError as e:
        raise ImportError(
            "anthropic_llm requires the anthropic package. "
            "Install with: pip install aevyra-origin[anthropic]"
        ) from e

    client = Anthropic(api_key=api_key) if api_key else Anthropic()
    return _AnthropicLLM(client, model, max_tokens, temperature)


def openai_llm(
    model: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    max_tokens: int = 16384,
    temperature: float = 0.0,
) -> _OpenAILLM:
    """Factory: any OpenAI-compatible endpoint.

    Works with OpenAI, OpenRouter, Together, Groq, DeepInfra, Ollama's
    OpenAI shim, or any locally-hosted OpenAI-compatible server.

    Requires the ``openai`` extra::

        pip install aevyra-origin[openai]

    Args:
        model:       Model ID as the endpoint expects it (e.g. ``"gpt-4o"``
                     or ``"qwen/qwen3-8b"``).
        api_key:     API key. If None, the OpenAI SDK picks up
                     ``OPENAI_API_KEY`` from the environment.
        base_url:    Optional custom endpoint (e.g. ``"https://openrouter.ai/api/v1"``
                     or ``"http://localhost:11434/v1"`` for Ollama).
        max_tokens:  Max output tokens per call.
        temperature: Sampling temperature. Defaults to 0.0.

    Returns:
        A callable ``(prompt: str) -> str`` with a ``tokens_used`` attribute.
        Satisfies :data:`LLMFn`.
    """
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError(
            "openai_llm requires the openai package. "
            "Install with: pip install aevyra-origin[openai]"
        ) from e

    kwargs: dict = {}
    if api_key is not None:
        kwargs["api_key"] = api_key
    if base_url is not None:
        kwargs["base_url"] = base_url
    client = OpenAI(**kwargs)
    return _OpenAILLM(client, model, max_tokens, temperature)


__all__ = ["LLMFn", "anthropic_llm", "openai_llm"]
