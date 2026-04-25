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

"""LLM-as-critic attribution method.

Reference-free. One LLM call. The LLM reads the rubric, the score, the
ideal (if any), and the full execution trace, and returns a ranked list
of culprit spans with severity, confidence, and reasoning.

This is Origin's fastest and most general method. It works with any
trace — linear or DAG — with any rubric, and makes no assumptions about
the judge being decomposable. The output maps directly onto the
``Attribution`` type.

For DAG traces the critic attributes at the **span** level — a specific
call site. Callers that want to know which prompt to optimize (for
Reflex) should read :py:meth:`Attribution.by_prompt` on the result.
"""

from __future__ import annotations

import logging
from typing import Any

from aevyra_witness import AgentTrace

from aevyra_origin._json import JSONParseError, extract_json
from aevyra_origin.llm import LLMFn
from aevyra_origin.prompts import format_critic_prompt
from aevyra_origin.result import NodeAttribution, VALID_SEVERITIES

logger = logging.getLogger(__name__)


class CriticError(RuntimeError):
    """Raised when the critic LLM returns an unusable response."""


def run_critic(
    *,
    trace: AgentTrace,
    score: float,
    rubric: str,
    llm: LLMFn,
) -> dict[str, Any]:
    """Run the LLM-as-critic method and return a raw parsed result.

    The result is a dict with:
        - ``summary``:  str
        - ``culprits``: list[NodeAttribution]
        - ``raw``:      the LLM's unparsed response string

    Args:
        trace:  Execution trace to diagnose.
        score:  Judge score being explained.
        rubric: The judge criteria (as passed to Verdict).
        llm:    Any callable ``(prompt: str) -> str``.

    Raises:
        CriticError: if the LLM response cannot be parsed or references
                     spans that do not exist in the trace.
    """
    prompt = format_critic_prompt(
        rubric=rubric.strip(),
        score=_fmt_score(score),
        ideal=trace.ideal if trace.ideal is not None else "<not provided>",
        trace_text=trace.to_trace_text(),
    )

    raw = llm(prompt)

    try:
        parsed = extract_json(raw)
    except JSONParseError as e:
        raise CriticError(f"critic response was not parseable JSON: {e}") from e

    summary, culprits = _normalize(parsed, trace)

    return {
        "summary": summary,
        "culprits": culprits,
        "raw": raw,
    }


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _fmt_score(score: float) -> str:
    """Format a score so the LLM sees a clean, stable number."""
    if isinstance(score, bool):
        return "1.000" if score else "0.000"
    try:
        return f"{float(score):.3f}"
    except (TypeError, ValueError):
        return str(score)


def _normalize(
    parsed: dict[str, Any],
    trace: AgentTrace,
) -> tuple[str, list[NodeAttribution]]:
    """Turn a parsed JSON blob into (summary, culprits).

    Tolerant of minor malformedness: missing reasoning becomes ``""``,
    clamped confidences outside [0, 1] are snapped in, unknown severities
    raise ``CriticError``. Culprit resolution is strict:

    - If ``node_id`` is provided, it must exist in the trace.
    - Else, ``node_name`` must exist AND be unique in the trace — when
      names repeat, ``node_id`` is required.
    - ``node_name`` (when provided alongside ``node_id``) should match
      the span's actual name; a mismatch is a warning (logged) but not
      fatal — ``node_id`` is authoritative.
    """
    summary = str(parsed.get("summary", "")).strip()
    raw_culprits = parsed.get("culprits", [])
    if not isinstance(raw_culprits, list):
        raise CriticError(f"'culprits' must be a list, got {type(raw_culprits).__name__}")

    # Precompute name ambiguity. If a name appears more than once, node_id
    # is required to disambiguate.
    name_counts: dict[str, int] = {}
    for n in trace.nodes:
        name_counts[n.name] = name_counts.get(n.name, 0) + 1

    culprits: list[NodeAttribution] = []
    for i, c in enumerate(raw_culprits):
        if not isinstance(c, dict):
            raise CriticError(f"culprit #{i} is not an object: {c!r}")

        node_id = str(c.get("node_id", "")).strip() or None
        node_name = str(c.get("node_name", "")).strip()

        node = _resolve_span(
            trace=trace,
            node_id=node_id,
            node_name=node_name,
            name_counts=name_counts,
            index=i,
        )

        severity = str(c.get("severity", "")).strip().lower()
        if severity not in VALID_SEVERITIES:
            raise CriticError(
                f"culprit #{i} has invalid severity {severity!r}; must be one of {VALID_SEVERITIES}"
            )

        confidence = _coerce_float(c.get("confidence"))
        if confidence is None:
            raise CriticError(f"culprit #{i} has non-numeric confidence {c.get('confidence')!r}")
        confidence = _clamp(confidence, 0.0, 1.0)

        reasoning = str(c.get("reasoning", "")).strip()

        culprits.append(
            NodeAttribution(
                node_name=node.name,
                severity=severity,  # type: ignore[arg-type]
                confidence=confidence,
                reasoning=reasoning,
                node_id=node.id or None,
                prompt_id=node.prompt_id,
            )
        )

    # Sort by confidence descending.
    culprits.sort(key=lambda n: n.confidence, reverse=True)
    return summary, culprits


def _resolve_span(
    *,
    trace: AgentTrace,
    node_id: str | None,
    node_name: str,
    name_counts: dict[str, int],
    index: int,
):
    """Resolve an LLM-emitted culprit to an actual span in the trace.

    Prefers ``node_id`` when given. Falls back to name when names are
    unique. Raises ``CriticError`` if neither resolves cleanly — we
    intentionally refuse to guess.
    """
    # Prefer id.
    if node_id:
        node = trace.by_id(node_id)
        if node is None:
            valid = [n.id for n in trace.nodes]
            raise CriticError(
                f"culprit #{index} references unknown node_id {node_id!r}; trace has ids {valid}"
            )
        if node_name and node_name != node.name:
            logger.warning(
                "culprit #%d: node_name %r does not match span name %r for id=%s; "
                "using id as authoritative",
                index,
                node_name,
                node.name,
                node_id,
            )
        return node

    # No id — fall back to name, but only if unambiguous.
    if not node_name:
        raise CriticError(f"culprit #{index} is missing both 'node_id' and 'node_name'")

    count = name_counts.get(node_name, 0)
    if count == 0:
        names = sorted(set(n.name for n in trace.nodes))
        raise CriticError(
            f"culprit #{index} references unknown node_name {node_name!r}; trace has names {names}"
        )
    if count > 1:
        matching_ids = [n.id for n in trace.nodes if n.name == node_name]
        raise CriticError(
            f"culprit #{index} references ambiguous node_name {node_name!r} "
            f"(trace has {count} spans with this name: ids {matching_ids}); "
            f"include 'node_id' to disambiguate"
        )

    # Unique by name.
    for n in trace.nodes:
        if n.name == node_name:
            return n
    # Unreachable.
    raise CriticError(f"culprit #{index}: internal error resolving span {node_name!r}")


def _coerce_float(v: Any) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


__all__ = ["CriticError", "run_critic"]
