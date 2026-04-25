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

"""Score-decomposition attribution method.

Reference-free. One LLM call. The LLM reads the rubric and the trace,
enumerates the rubric's underlying criteria, and attributes each
criterion to the span(s) responsible. Per-span blame is then aggregated
across failed criteria to produce an ``Attribution``.

Where LLM-as-critic asks "who failed?", decomposition asks "what failed
and whose job was that?". The two are complementary — critic tends to
identify the most egregious single cause, decomposition tends to surface
distributed failures that critic would collapse into one.

For DAG traces the aggregation keys on ``node_id`` rather than name, so
two spans with the same name (e.g. a planner firing at steps 1 and 2)
are blamed separately. Callers that want prompt-level rollup should use
:py:meth:`Attribution.by_prompt`.
"""

from __future__ import annotations

import logging
from typing import Any

from aevyra_witness import AgentTrace, TraceNode

from aevyra_origin._json import JSONParseError, extract_json
from aevyra_origin.llm import LLMFn
from aevyra_origin.prompts import format_decomposition_prompt
from aevyra_origin.result import NodeAttribution

logger = logging.getLogger(__name__)


class DecompositionError(RuntimeError):
    """Raised when the decomposition LLM returns an unusable response."""


def run_decomposition(
    *,
    trace: AgentTrace,
    score: float,
    rubric: str,
    llm: LLMFn,
) -> dict[str, Any]:
    """Run the score-decomposition method and return a raw parsed result.

    The result is a dict with:
        - ``summary``:  str  (synthesized from the criteria analysis)
        - ``culprits``: list[NodeAttribution]
        - ``criteria``: the parsed per-criterion attribution (for debugging)
        - ``raw``:      the LLM's unparsed response string

    Raises:
        DecompositionError: on unparseable or malformed responses.
    """
    prompt = format_decomposition_prompt(
        rubric=rubric.strip(),
        score=_fmt_score(score),
        ideal=trace.ideal if trace.ideal is not None else "<not provided>",
        trace_text=trace.to_trace_text(),
    )

    raw = llm(prompt)

    try:
        parsed = extract_json(raw)
    except JSONParseError as e:
        raise DecompositionError(f"decomposition response was not parseable JSON: {e}") from e

    criteria = _parse_criteria(parsed, trace)
    culprits = _aggregate(criteria, trace)
    summary = _summarize(criteria)

    return {
        "summary": summary,
        "culprits": culprits,
        "criteria": criteria,
        "raw": raw,
    }


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _parse_criteria(parsed: dict[str, Any], trace: AgentTrace) -> list[dict[str, Any]]:
    raw_criteria = parsed.get("criteria", [])
    if not isinstance(raw_criteria, list):
        raise DecompositionError(f"'criteria' must be a list, got {type(raw_criteria).__name__}")

    # Precompute name ambiguity.
    name_counts: dict[str, int] = {}
    for n in trace.nodes:
        name_counts[n.name] = name_counts.get(n.name, 0) + 1

    out: list[dict[str, Any]] = []
    for i, c in enumerate(raw_criteria):
        if not isinstance(c, dict):
            raise DecompositionError(f"criterion #{i} is not an object: {c!r}")

        description = str(c.get("criterion", "")).strip()
        if not description:
            raise DecompositionError(f"criterion #{i} is missing 'criterion' description")

        satisfied = _coerce_bool(c.get("satisfied"))
        if satisfied is None:
            raise DecompositionError(
                f"criterion #{i} ({description!r}) has non-boolean 'satisfied': "
                f"{c.get('satisfied')!r}"
            )

        raw_nodes = c.get("nodes", [])
        if not isinstance(raw_nodes, list):
            raise DecompositionError(
                f"criterion #{i} 'nodes' must be a list, got {type(raw_nodes).__name__}"
            )

        nodes: list[dict[str, Any]] = []
        for j, n in enumerate(raw_nodes):
            if not isinstance(n, dict):
                raise DecompositionError(f"criterion #{i} node #{j} is not an object: {n!r}")
            node_id = str(n.get("node_id", "")).strip() or None
            node_name = str(n.get("node_name", "")).strip()

            span = _resolve_span(
                trace=trace,
                node_id=node_id,
                node_name=node_name,
                name_counts=name_counts,
                crit_index=i,
                node_index=j,
            )

            contribution = _coerce_float(n.get("contribution"))
            if contribution is None:
                raise DecompositionError(
                    f"criterion #{i} node #{j} has non-numeric contribution "
                    f"{n.get('contribution')!r}"
                )
            contribution = max(0.0, min(1.0, contribution))
            reasoning = str(n.get("reasoning", "")).strip()
            nodes.append(
                {
                    "node_id": span.id,
                    "node_name": span.name,
                    "prompt_id": span.prompt_id,
                    "contribution": contribution,
                    "reasoning": reasoning,
                }
            )

        # Normalize contributions to sum to 1.0 if the LLM's sum is off.
        total = sum(n["contribution"] for n in nodes)
        if total > 0 and abs(total - 1.0) > 0.01:
            for n in nodes:
                n["contribution"] = n["contribution"] / total

        out.append(
            {
                "criterion": description,
                "satisfied": satisfied,
                "nodes": nodes,
            }
        )
    return out


def _aggregate(
    criteria: list[dict[str, Any]],
    trace: AgentTrace,
) -> list[NodeAttribution]:
    """Aggregate per-criterion contributions into per-span blame.

    For each failed criterion, each responsible span accrues blame
    equal to its ``contribution`` weight. Totals are divided by the
    number of failed criteria so the final score is in [0, 1] per span
    (a span that gets full blame on every failed criterion lands at
    1.0).

    Aggregation keys on ``node_id`` (unique) rather than ``node_name``
    (may repeat in DAG traces).

    Severity is assigned by thresholding the final blame score:
        >= 0.5   → "primary"
        >= 0.2   → "contributing"
        > 0      → "minor"
    """
    failed = [c for c in criteria if not c["satisfied"]]
    if not failed:
        return []

    # Per-span blame / reasoning, keyed on node_id.
    blame: dict[str, float] = {}
    reasons: dict[str, list[str]] = {}
    span_lookup: dict[str, TraceNode] = {n.id: n for n in trace.nodes}

    for c in failed:
        for n in c["nodes"]:
            key = n["node_id"]
            blame[key] = blame.get(key, 0.0) + n["contribution"]
            if n["reasoning"]:
                reasons.setdefault(key, []).append(f"[{c['criterion']}] {n['reasoning']}")

    n_failed = len(failed)
    # Preserve trace order for ties.
    order = {node.id: idx for idx, node in enumerate(trace.nodes)}

    culprits: list[NodeAttribution] = []
    for node_id, total in blame.items():
        score = min(1.0, total / n_failed)
        if score <= 0:
            continue
        if score >= 0.5:
            severity = "primary"
        elif score >= 0.2:
            severity = "contributing"
        else:
            severity = "minor"
        span = span_lookup.get(node_id)
        if span is None:
            # Shouldn't happen — spans were validated during parse — but
            # degrade gracefully rather than crashing aggregation.
            continue
        culprits.append(
            NodeAttribution(
                node_name=span.name,
                severity=severity,  # type: ignore[arg-type]
                confidence=score,
                reasoning="  ".join(reasons.get(node_id, [])) or "(no per-criterion reasoning)",
                node_id=span.id or None,
                prompt_id=span.prompt_id,
            )
        )

    culprits.sort(key=lambda n: (-n.confidence, order.get(n.node_id or "", 1e9)))
    return culprits


def _summarize(criteria: list[dict[str, Any]]) -> str:
    """One-paragraph synthesis of the criteria analysis.

    Kept intentionally concise — ``Attribution.summary`` is for quick
    orientation; detail lives in the per-span reasoning and ``raw``.
    """
    if not criteria:
        return "No criteria were enumerated."
    passed = sum(1 for c in criteria if c["satisfied"])
    failed = len(criteria) - passed
    if failed == 0:
        return f"All {len(criteria)} criteria passed."
    parts = [
        f"Score decomposition: {passed}/{len(criteria)} criteria passed, {failed} failed.",
    ]
    for c in criteria:
        if not c["satisfied"]:
            contributors = (
                ", ".join(
                    f"{n['node_name']}({n['node_id']}) [{n['contribution']:.2f}]"
                    for n in c["nodes"]
                )
                or "no spans attributed"
            )
            parts.append(f"Failed: {c['criterion']} — {contributors}.")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Span resolution (shared shape with critic._resolve_span, but local copy
# to keep the two methods independent — they may diverge later)
# ---------------------------------------------------------------------------


def _resolve_span(
    *,
    trace: AgentTrace,
    node_id: str | None,
    node_name: str,
    name_counts: dict[str, int],
    crit_index: int,
    node_index: int,
) -> TraceNode:
    if node_id:
        node = trace.by_id(node_id)
        if node is None:
            valid = [n.id for n in trace.nodes]
            raise DecompositionError(
                f"criterion #{crit_index} node #{node_index} references unknown "
                f"node_id {node_id!r}; trace has ids {valid}"
            )
        if node_name and node_name != node.name:
            logger.warning(
                "criterion #%d node #%d: node_name %r does not match span name %r for id=%s",
                crit_index,
                node_index,
                node_name,
                node.name,
                node_id,
            )
        return node

    if not node_name:
        raise DecompositionError(
            f"criterion #{crit_index} node #{node_index} is missing both 'node_id' and 'node_name'"
        )

    count = name_counts.get(node_name, 0)
    if count == 0:
        names = sorted(set(n.name for n in trace.nodes))
        raise DecompositionError(
            f"criterion #{crit_index} node #{node_index} references unknown "
            f"node_name {node_name!r}; trace has names {names}"
        )
    if count > 1:
        matching_ids = [n.id for n in trace.nodes if n.name == node_name]
        raise DecompositionError(
            f"criterion #{crit_index} node #{node_index} references ambiguous "
            f"node_name {node_name!r} (trace has {count} spans: ids {matching_ids}); "
            f"include 'node_id' to disambiguate"
        )

    for n in trace.nodes:
        if n.name == node_name:
            return n
    raise DecompositionError(
        f"criterion #{crit_index} node #{node_index}: internal error resolving span {node_name!r}"
    )


# ---------------------------------------------------------------------------
# Coercion helpers
# ---------------------------------------------------------------------------


def _fmt_score(score: float) -> str:
    if isinstance(score, bool):
        return "1.000" if score else "0.000"
    try:
        return f"{float(score):.3f}"
    except (TypeError, ValueError):
        return str(score)


def _coerce_float(v: Any) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _coerce_bool(v: Any) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "yes", "1"):
            return True
        if s in ("false", "no", "0"):
            return False
    return None


__all__ = ["DecompositionError", "run_decomposition"]
