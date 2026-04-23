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

"""Judge adapters — turn third-party scorers into Origin :data:`Judge` callables.

Origin's ``Judge`` contract is deliberately narrow::

    Judge = Callable[[AgentTrace], float]

Anything that scores a trace is a judge. Third-party evaluators usually
have a different shape — Verdict's ``Metric.score(response, ideal, messages)``
returns a ``ScoreResult``, for example. This module provides adapters
that bridge the gap without forcing Origin to import (or depend on) any
specific evaluator.

The :func:`judge_from_verdict` adapter accepts any object that
duck-types Verdict's ``Metric.score(...)`` signature. No top-level
import of ``aevyra_verdict`` — Verdict stays an optional dependency.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from aevyra_witness import AgentTrace

from aevyra_origin.ablation import Judge

__all__ = ["judge_from_verdict"]


def judge_from_verdict(
    metric: Any,
    *,
    extract_response: Callable[[AgentTrace], str] | None = None,
    extract_messages: Callable[[AgentTrace], list[dict[str, str]] | None] | None = None,
    ideal: str | None = None,
) -> Judge:
    """Turn a Verdict ``Metric`` into an Origin :data:`Judge`.

    Origin's attribution methods care only about the numeric score; the
    judge's internal reasoning lives on ``ScoreResult.reasoning`` and is
    ignored by the attribution engine (but surfaced in ``Attribution.raw``
    when the judge is invoked).

    Args:
        metric:            Any object exposing
                           ``.score(response: str, ideal: str | None,
                           messages: list[dict] | None) -> ScoreResult``.
                           Verdict's ``LLMJudge``, ``ExactMatch``,
                           ``BleuScore``, ``RougeScore``, and custom
                           metrics all match this shape.
        extract_response:  How to pull the agent's final output out of
                           the trace, as a string. Defaults to
                           :func:`default_response_from_trace`, which
                           returns the last root span's output serialized
                           to a string.
        extract_messages:  How to reconstruct the original user-facing
                           conversation from the trace. Defaults to
                           :func:`default_messages_from_trace`, which
                           wraps the first root span's input as a single
                           user message. Return ``None`` to omit
                           conversation context.
        ideal:             Override the reference output. Defaults to
                           ``trace.ideal`` at judge-call time.

    Returns:
        A ``Judge`` callable compatible with ``diagnose_pipeline``,
        ``Origin(runner=..., judge=...)``, and any other Origin API that
        accepts a judge.

    Example::

        from aevyra_verdict import LLMJudge
        from aevyra_verdict.providers import get_provider
        from aevyra_origin.judges import judge_from_verdict

        metric = LLMJudge(judge_provider=get_provider("anthropic"))
        judge = judge_from_verdict(metric)
        score = judge(my_trace)  # -> float in [0, 1]
    """
    _extract_response = extract_response or default_response_from_trace
    _extract_messages = extract_messages or default_messages_from_trace

    def judge(trace: AgentTrace) -> float:
        response = _extract_response(trace)
        messages = _extract_messages(trace)
        use_ideal = ideal if ideal is not None else trace.ideal
        result = metric.score(response=response, ideal=use_ideal, messages=messages)
        # Duck-typed ScoreResult: prefer .score attribute; fall back to
        # treating the return value itself as a number.
        raw = getattr(result, "score", result)
        try:
            return float(raw)
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"judge_from_verdict: metric.score() returned {raw!r} which is "
                f"not coercible to float"
            ) from e

    return judge


# ---------------------------------------------------------------------------
# Default extractors — exported so users can wrap them
# ---------------------------------------------------------------------------


def default_response_from_trace(trace: AgentTrace) -> str:
    """Pull the agent's final output out of a trace as a string.

    Picks the last root span (executed last among the top level of the
    DAG) and returns its ``output`` as a string. Non-string outputs are
    JSON-encoded for judge consumption.

    Raises ``ValueError`` if the trace has no root spans.
    """
    roots = trace.roots
    if not roots:
        raise ValueError(
            "trace has no root span; cannot extract a response. "
            "Supply extract_response=... to judge_from_verdict()."
        )
    last = roots[-1]
    out = last.output
    if out is None:
        return ""
    if isinstance(out, str):
        return out
    try:
        return json.dumps(out, ensure_ascii=False, default=str)
    except TypeError:
        return repr(out)


def default_messages_from_trace(
    trace: AgentTrace,
) -> list[dict[str, str]] | None:
    """Reconstruct a simple ``messages`` list for a Verdict judge.

    Uses the first root span's ``input`` as the user message. Returns
    ``None`` (not a list) when the input isn't a string — Verdict judges
    handle ``messages=None`` cleanly.
    """
    roots = trace.roots
    if not roots:
        return None
    inp = roots[0].input
    if isinstance(inp, str):
        return [{"role": "user", "content": inp}]
    return None
