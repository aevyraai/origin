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

"""Ablation attribution — the causal second opinion.

Ablation is Origin's only method that makes *causal* claims. The
LLM-as-critic and score-decomposition methods are pattern matchers: they
read the trace and tell you what probably went wrong. Ablation actually
perturbs the run — for each candidate span, it re-executes the pipeline
with that span's output replaced by a neutral placeholder and re-scores
the result. The change in judge score is the span's causal contribution.

This makes ablation the natural cross-check against the LLM methods. If
the critic blames span ``p2`` but ablating ``p2`` leaves the score
unchanged, the critic is wrong. If ablating ``t1b`` drops the score by
0.5 but neither LLM method mentioned it, the LLM methods missed the
real culprit.

Ablation is more expensive than the LLM methods: one pipeline re-run
per candidate span, plus one judge call per re-run. Use ``budget`` or
``candidates`` to cap cost.

Runner contract
---------------

Origin itself never executes the user's pipeline. Ablation requires the
caller to supply a ``runner`` callable that knows how::

    runner(trace: AgentTrace, overrides: dict[str, Any]) -> AgentTrace

``overrides`` maps span id → forced output value. The runner must
replay the pipeline from the beginning, and **whenever** it would
execute a span whose id appears in ``overrides``, it must skip the real
execution and use the forced value instead. Spans downstream of the
overridden span re-execute against the forced value as if that were
the real output.

The runner is expected to be deterministic. Real pipelines have side
effects (LLM calls, HTTP requests, stateful tools) — the caller is
responsible for caching or mocking those so repeated runs with the
same inputs produce the same outputs. Without determinism, score
deltas reflect noise rather than span contribution.

Judge contract
--------------

``judge(trace: AgentTrace) -> float`` returns the score for an
(ablated) trace. Typically wraps the same Verdict judge that produced
the original score. The ``score_range`` argument to :func:`run_ablation`
tells the normalizer how to convert a raw delta into a confidence.

Placeholder strategies
----------------------

Two strategies are supported, answering different questions:

- ``"null"``: replace the span's output with a neutral value (``None``,
  ``""``, or ``[]`` depending on the original type). Answers "did this
  span contribute any signal?"
- ``"ideal"``: replace with the trace's ``ideal`` output. Answers
  "would a perfect output here have saved the run?" Requires the
  trace to have an ``ideal`` set; falls back to ``"null"`` otherwise
  with a logged warning.

For most debugging workflows ``"null"`` is the right default — it
identifies which spans carry load. ``"ideal"`` is useful for asking
"if I fix this one span, does the pipeline succeed?" which is more of
an optimization-planning question.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Literal

from aevyra_witness import AgentTrace, TraceNode

from aevyra_origin.result import NodeAttribution

logger = logging.getLogger(__name__)


# Public protocol aliases. Deliberately Callable rather than Protocol so
# the user can pass a lambda or a bound method without ceremony.
Runner = Callable[[AgentTrace, dict[str, Any]], AgentTrace]
"""A pipeline-replay callable.

Takes the original trace and a mapping of ``span_id → forced_output``,
returns a new trace representing the re-execution with those outputs
forced. See the module docstring for the behavioral contract.
"""

Judge = Callable[[AgentTrace], float]
"""A trace-scoring callable, typically wrapping Verdict."""

Placeholder = Literal["null", "ideal"]
"""Supported ablation placeholder strategies. See module docstring."""

VALID_PLACEHOLDERS: tuple[str, ...] = ("null", "ideal")


# Delta-to-severity thresholds. Matches the decomposition method's
# convention so users get consistent severity semantics across methods.
_SEVERITY_PRIMARY = 0.5
_SEVERITY_CONTRIBUTING = 0.2


class AblationError(RuntimeError):
    """Raised when ablation cannot proceed.

    Reasons include: the runner returned a non-``AgentTrace`` value,
    the judge returned a non-numeric score, or every candidate span
    failed to execute (a total failure is worth surfacing rather than
    silently returning no culprits).
    """


# ---------------------------------------------------------------------------
# Per-span result structure — internal, flattened into NodeAttribution on exit
# ---------------------------------------------------------------------------


@dataclass
class _AblationEffect:
    """Internal record of one span's ablation outcome."""

    node: TraceNode
    ablated_score: float
    raw_delta: float  # original - ablated; positive means span helped
    normalized_delta: float  # |raw_delta| / score_range
    direction: str  # "helpful" | "harmful" — signed from raw_delta
    placeholder_used: Any
    error: str | None = None  # runner/judge error, if any


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_ablation(
    *,
    trace: AgentTrace,
    score: float,
    runner: Runner,
    judge: Judge,
    rubric: str = "",
    placeholder: Placeholder = "null",
    candidates: list[str] | None = None,
    budget: int | None = None,
    min_delta: float = 0.05,
    score_range: tuple[float, float] = (0.0, 1.0),
) -> dict[str, Any]:
    """Run ablation attribution over a trace.

    For each candidate span, re-runs the pipeline with that span's
    output forcibly replaced by a neutral placeholder and re-scores the
    result. The change in judge score is the span's causal contribution.

    Args:
        trace:         The execution trace to diagnose.
        score:         Original judge score on the trace. Used as the
                       reference when computing deltas; also returned in
                       per-span reasoning strings. **Must** be the score
                       produced by ``judge(trace)`` on the same judge,
                       otherwise deltas are meaningless.
        runner:        Pipeline-replay callable. See module docstring.
        judge:         Trace-scoring callable. See module docstring.
        rubric:        Accepted for API parity with the other methods;
                       unused by ablation itself (the rubric is baked
                       into the judge).
        placeholder:   ``"null"`` replaces outputs with a neutral value;
                       ``"ideal"`` replaces with ``trace.ideal`` (falls
                       back to ``"null"`` if ``ideal`` is ``None``).
        candidates:    Explicit span ids to ablate. If ``None``, every
                       span in the trace is a candidate.
        budget:        Upper bound on the number of ablation runs. When
                       set, candidates are taken in trace order until the
                       budget is exhausted.
        min_delta:     Skip any span whose normalized absolute delta
                       falls below this threshold. Default 0.05 — noise
                       floor for spans whose output doesn't move the
                       score.
        score_range:   ``(lo, hi)`` range the judge scores in. Used to
                       normalize deltas to ``[0, 1]`` confidences.
                       Defaults to ``(0.0, 1.0)`` — the Verdict convention.

    Returns:
        A dict mirroring the shape returned by :func:`run_critic` and
        :func:`run_decomposition`::

            {
                "summary":   str,
                "culprits":  list[NodeAttribution],
                "effects":   list[dict],   # per-candidate diagnostic trail
                "original_score": float,
                "placeholder":    str,
                "num_candidates": int,
                "num_effects":    int,
                "num_failed":     int,
            }

        Culprits are ranked by absolute normalized delta descending. The
        ``effects`` list includes every candidate that was tried — even
        those below ``min_delta`` — so callers can audit the search.

    Raises:
        AblationError: runner/judge contract violations, or every
                       candidate failed (``num_failed == num_candidates``
                       and no culprits).
        ValueError:    invalid arguments (bad placeholder, empty trace,
                       non-positive score_range, candidates referencing
                       unknown spans).
    """
    # --- Argument validation -------------------------------------------------
    if not trace.nodes:
        raise ValueError("cannot ablate a trace with zero nodes")
    if placeholder not in VALID_PLACEHOLDERS:
        raise ValueError(
            f"placeholder must be one of {VALID_PLACEHOLDERS}, got {placeholder!r}"
        )
    if not callable(runner):
        raise TypeError(f"runner must be callable, got {type(runner).__name__}")
    if not callable(judge):
        raise TypeError(f"judge must be callable, got {type(judge).__name__}")
    lo, hi = score_range
    span = float(hi) - float(lo)
    if span <= 0:
        raise ValueError(
            f"score_range must have hi > lo; got {score_range!r}"
        )

    # --- Candidate selection -------------------------------------------------
    chosen = _select_candidates(trace, candidates, budget)
    if not chosen:
        # No candidates is not an error — it's a legitimate "nothing to ablate"
        # result (e.g. budget=0, or user-supplied empty list).
        return {
            "summary": "No candidate spans were ablated.",
            "culprits": [],
            "effects": [],
            "original_score": float(score),
            "placeholder": placeholder,
            "num_candidates": 0,
            "num_effects": 0,
            "num_failed": 0,
        }

    # --- Placeholder fallback check -----------------------------------------
    effective_placeholder: Placeholder = placeholder
    if placeholder == "ideal" and trace.ideal is None:
        logger.warning(
            "ablation: placeholder='ideal' requested but trace.ideal is None; "
            "falling back to 'null'"
        )
        effective_placeholder = "null"

    # --- Per-span ablation loop ---------------------------------------------
    effects: list[_AblationEffect] = []
    num_failed = 0
    for node in chosen:
        forced = _build_placeholder(node, effective_placeholder, trace)
        try:
            ablated_trace = runner(trace, {node.id: forced})
        except Exception as exc:  # runner is user code; isolate failures
            logger.warning(
                "ablation: runner failed for span %s (id=%s): %s",
                node.name,
                node.id,
                exc,
            )
            num_failed += 1
            effects.append(
                _AblationEffect(
                    node=node,
                    ablated_score=float("nan"),
                    raw_delta=0.0,
                    normalized_delta=0.0,
                    direction="unknown",
                    placeholder_used=forced,
                    error=f"runner failed: {exc}",
                )
            )
            continue

        if not isinstance(ablated_trace, AgentTrace):
            raise AblationError(
                f"runner returned {type(ablated_trace).__name__}, expected AgentTrace "
                f"(while ablating span name={node.name!r}, id={node.id!r})"
            )

        try:
            ablated_score = judge(ablated_trace)
        except Exception as exc:
            logger.warning(
                "ablation: judge failed for ablated span %s (id=%s): %s",
                node.name,
                node.id,
                exc,
            )
            num_failed += 1
            effects.append(
                _AblationEffect(
                    node=node,
                    ablated_score=float("nan"),
                    raw_delta=0.0,
                    normalized_delta=0.0,
                    direction="unknown",
                    placeholder_used=forced,
                    error=f"judge failed: {exc}",
                )
            )
            continue

        if not isinstance(ablated_score, (int, float)) or isinstance(ablated_score, bool):
            raise AblationError(
                f"judge returned {type(ablated_score).__name__}, expected numeric "
                f"(while ablating span name={node.name!r}, id={node.id!r})"
            )

        raw_delta = float(score) - float(ablated_score)
        normalized = abs(raw_delta) / span
        direction = "helpful" if raw_delta > 0 else ("harmful" if raw_delta < 0 else "neutral")
        effects.append(
            _AblationEffect(
                node=node,
                ablated_score=float(ablated_score),
                raw_delta=raw_delta,
                normalized_delta=normalized,
                direction=direction,
                placeholder_used=forced,
            )
        )

    # --- Total-failure guard ------------------------------------------------
    if num_failed == len(chosen) and chosen:
        raise AblationError(
            f"all {num_failed} ablation runs failed; see warnings for per-span "
            "errors. Check the runner's determinism and the span ids it accepts."
        )

    # --- Build NodeAttribution culprits from effects above threshold --------
    culprits: list[NodeAttribution] = []
    for eff in effects:
        if eff.error is not None:
            continue
        if eff.normalized_delta < min_delta:
            continue
        culprits.append(_to_node_attribution(eff, score, effective_placeholder))

    culprits.sort(key=lambda c: c.confidence, reverse=True)

    summary = _build_summary(
        effects=effects,
        culprits=culprits,
        original_score=float(score),
        placeholder=effective_placeholder,
        num_failed=num_failed,
        min_delta=min_delta,
    )

    return {
        "summary": summary,
        "culprits": culprits,
        "effects": [_effect_to_dict(e) for e in effects],
        "original_score": float(score),
        "placeholder": effective_placeholder,
        "num_candidates": len(chosen),
        "num_effects": len(effects),
        "num_failed": num_failed,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _select_candidates(
    trace: AgentTrace,
    candidates: list[str] | None,
    budget: int | None,
) -> list[TraceNode]:
    """Resolve user-supplied candidate ids (or default) and apply budget.

    Unknown candidate ids raise ``ValueError`` — failing loudly here is
    better than silently skipping, because a mistyped id looks like a
    legitimate "no signal" result otherwise.
    """
    if candidates is None:
        chosen = list(trace.nodes)
    else:
        chosen = []
        unknown: list[str] = []
        for sid in candidates:
            n = trace.by_id(sid)
            if n is None:
                unknown.append(sid)
            else:
                chosen.append(n)
        if unknown:
            raise ValueError(
                f"candidates reference unknown span ids: {unknown!r}"
            )

    if budget is not None:
        if budget < 0:
            raise ValueError(f"budget must be non-negative, got {budget}")
        chosen = chosen[:budget]
    return chosen


def _build_placeholder(
    node: TraceNode,
    placeholder: Placeholder,
    trace: AgentTrace,
) -> Any:
    """Construct the forced output value for one ablation run.

    The ``"null"`` strategy uses the zero value for the span's output
    type — empty string for strings, empty list/dict for containers,
    ``None`` otherwise. This preserves downstream type-checking (a
    consumer expecting a list won't crash on ``None``) while removing
    the span's signal. The ``"ideal"`` strategy uses the trace-level
    ideal output.
    """
    if placeholder == "ideal":
        # Caller already confirmed ideal is set at the run_ablation level;
        # be defensive here in case someone calls this helper directly.
        if trace.ideal is not None:
            return trace.ideal
    # null-style: preserve output shape, drop content.
    v = node.output
    if isinstance(v, str):
        return ""
    if isinstance(v, list):
        return []
    if isinstance(v, dict):
        return {}
    if isinstance(v, tuple):
        return ()
    return None


def _to_node_attribution(
    eff: _AblationEffect,
    original_score: float,
    placeholder: Placeholder,
) -> NodeAttribution:
    """Turn an ablation effect into a ``NodeAttribution``."""
    confidence = min(1.0, max(0.0, eff.normalized_delta))
    if eff.normalized_delta >= _SEVERITY_PRIMARY:
        severity: Any = "primary"
    elif eff.normalized_delta >= _SEVERITY_CONTRIBUTING:
        severity = "contributing"
    else:
        severity = "minor"
    reasoning = _format_reasoning(eff, original_score, placeholder)
    return NodeAttribution(
        node_name=eff.node.name,
        severity=severity,
        confidence=confidence,
        reasoning=reasoning,
        node_id=eff.node.id or None,
        prompt_id=eff.node.prompt_id,
    )


def _format_reasoning(
    eff: _AblationEffect,
    original_score: float,
    placeholder: Placeholder,
) -> str:
    """One-paragraph explanation of what ablation showed for this span."""
    delta_sign = "+" if eff.raw_delta >= 0 else ""
    base = (
        f"Ablating this span (replacing its output via placeholder={placeholder!r}) "
        f"changed the judge score from {original_score:.3f} to {eff.ablated_score:.3f} "
        f"(delta={delta_sign}{eff.raw_delta:.3f}, normalized={eff.normalized_delta:.3f})."
    )
    if eff.direction == "helpful":
        verdict = (
            " The removal reduced the score, so this span is a material "
            "positive contributor — its real output was carrying load."
        )
    elif eff.direction == "harmful":
        verdict = (
            " The removal IMPROVED the score, so this span's real output is "
            "actively degrading the run. Consider fixing or removing it — "
            "its removal is a net win."
        )
    else:
        verdict = (
            " The score was unchanged; this span's output has no measurable "
            "effect on the judge."
        )
    return base + verdict


def _build_summary(
    *,
    effects: list[_AblationEffect],
    culprits: list[NodeAttribution],
    original_score: float,
    placeholder: Placeholder,
    num_failed: int,
    min_delta: float,
) -> str:
    """Short overview of what the ablation sweep showed."""
    if not effects:
        return "No spans were ablated."
    if not culprits:
        tested = len(effects) - num_failed
        return (
            f"Ablation tested {tested} span(s) with placeholder={placeholder!r}; "
            f"no span's removal moved the judge score by more than {min_delta:.2f}. "
            "Either the pipeline's signal is distributed across many spans below "
            "the threshold, or the judge does not discriminate at this resolution."
        )
    top = culprits[0]
    top_eff = next(e for e in effects if (e.node.id or e.node.name) == (top.node_id or top.node_name))
    harmful_count = sum(1 for e in effects if e.direction == "harmful" and e.error is None)
    parts = [
        f"Ablation (placeholder={placeholder!r}, original score={original_score:.3f}) "
        f"identified {len(culprits)} span(s) with material causal impact.",
        f"The highest-impact span is '{top.node_name}'"
        + (f" (id={top.node_id})" if top.node_id else "")
        + f": removing it moved the score to {top_eff.ablated_score:.3f} "
        f"(delta={top_eff.raw_delta:+.3f}).",
    ]
    if harmful_count:
        parts.append(
            f"{harmful_count} span(s) were found to be actively harmful "
            "(their removal improved the score)."
        )
    if num_failed:
        parts.append(f"{num_failed} ablation run(s) failed and were skipped.")
    return " ".join(parts)


def _effect_to_dict(eff: _AblationEffect) -> dict[str, Any]:
    """Serialize an ablation effect for the ``raw`` diagnostic trail."""
    return {
        "node_name": eff.node.name,
        "node_id": eff.node.id or None,
        "prompt_id": eff.node.prompt_id,
        "ablated_score": eff.ablated_score,
        "raw_delta": eff.raw_delta,
        "normalized_delta": eff.normalized_delta,
        "direction": eff.direction,
        "error": eff.error,
    }


__all__ = [
    "AblationError",
    "Judge",
    "Placeholder",
    "Runner",
    "VALID_PLACEHOLDERS",
    "run_ablation",
]
