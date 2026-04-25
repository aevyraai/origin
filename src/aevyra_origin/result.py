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

"""Result dataclasses for Origin attribution.

The attribution result is the core output of ``Origin.diagnose()``. It is
designed to be inspected in code, logged, serialized, and rendered in
CLIs or dashboards without loss of fidelity.

Origin attributes failure at the **span** level — one concrete execution
of a prompt or tool, identified by ``node_id`` in the trace. For DAG
traces where the same prompt fires at multiple steps, callers who want
to know which *prompt* to optimize can use :py:meth:`Attribution.by_prompt`
to roll span-level blame up to ``prompt_id``.

Conceptually::

    Attribution              The whole answer: "where did this fail?"
    ├── summary              One-paragraph human-readable overview
    ├── culprits             Ranked list of spans responsible
    │   └── NodeAttribution  Per-span: severity, confidence, reasoning,
    │                        optional node_id and prompt_id
    ├── method               Which methods ran ("critic", "decomposition", "all")
    ├── score                The input judge score being explained
    └── raw                  Method-level outputs, for debugging

    Attribution.by_prompt()  Roll span blame up to prompt_id → for Reflex
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

Severity = Literal["primary", "contributing", "minor"]


def _fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


VALID_SEVERITIES: tuple[str, ...] = ("primary", "contributing", "minor")

_SEVERITY_RANK = {"primary": 3, "contributing": 2, "minor": 1}


@dataclass
class NodeAttribution:
    """One span's share of the blame for a trace's failure.

    Args:
        node_name:  Human-readable name of the culprit span. Must match a
                    ``name`` that appears in the input trace. Names are
                    not unique in DAG traces — prefer ``node_id`` when
                    available for unambiguous identification.
        severity:   One of ``"primary"``, ``"contributing"``, ``"minor"``.
                    A trace typically has at most one primary culprit,
                    though compound failures may have more.
        confidence: Blame confidence, 0.0-1.0. How confident Origin is
                    that this span is responsible (orthogonal to severity).
        reasoning:  One-paragraph explanation grounded in the trace.
        node_id:    Unique span id from the trace. Required when the
                    trace has repeated node names (DAG / plan-act traces);
                    optional when names are unique.
        prompt_id:  Identity of the prompt behind this span, copied from
                    the trace. Enables rolling span-level blame up to
                    prompt-level for Reflex. ``None`` when the span has
                    no associated prompt (e.g. pure tool calls).
    """

    node_name: str
    severity: Severity
    confidence: float
    reasoning: str
    node_id: str | None = None
    prompt_id: str | None = None

    def __post_init__(self) -> None:
        if self.severity not in VALID_SEVERITIES:
            raise ValueError(f"severity must be one of {VALID_SEVERITIES}, got {self.severity!r}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_name": self.node_name,
            "severity": self.severity,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "node_id": self.node_id,
            "prompt_id": self.prompt_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "NodeAttribution":
        return cls(
            node_name=d["node_name"],
            severity=d["severity"],
            confidence=float(d["confidence"]),
            reasoning=d.get("reasoning", ""),
            node_id=d.get("node_id"),
            prompt_id=d.get("prompt_id"),
        )


@dataclass
class PromptAttribution:
    """Aggregated blame rolled up to the prompt level.

    When the same prompt fires at many call sites (e.g. a planner prompt
    at steps 1..N), Reflex cares about *the prompt* — updating it once
    affects every call site. ``PromptAttribution`` aggregates the
    per-span ``NodeAttribution``s that share a ``prompt_id``.

    Args:
        prompt_id:  The prompt identity from the trace.
        severity:   Max severity across the spans that share this prompt.
        confidence: Mean confidence across those spans (bounded to [0, 1]).
        spans:      The underlying per-span attributions that contributed
                    to this rollup, ordered by confidence descending.
        reasoning:  Concatenated reasoning from all contributing spans,
                    each prefixed with the span's node label.
    """

    prompt_id: str
    severity: Severity
    confidence: float
    spans: list[NodeAttribution]
    reasoning: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "severity": self.severity,
            "confidence": self.confidence,
            "spans": [s.to_dict() for s in self.spans],
            "reasoning": self.reasoning,
        }


@dataclass
class Attribution:
    """Full diagnostic result for one trace.

    Args:
        summary:   One-paragraph human-readable overview of why the trace failed.
        culprits:  Ranked list of ``NodeAttribution``, sorted by confidence
                   descending. Each culprit is a single span; DAG traces may
                   have multiple culprits sharing a ``prompt_id``.
        method:    The attribution method(s) used. One of ``"critic"``,
                   ``"decomposition"``, or ``"all"``.
        score:     The judge score being explained (typically 0.0-1.0).
        raw:       Method-level raw outputs for debugging.
    """

    summary: str
    culprits: list[NodeAttribution]
    method: str
    score: float
    raw: dict[str, Any] = field(default_factory=dict)
    llm_tokens: int = 0
    """Total LLM tokens consumed by the critic and decomposition methods."""
    ablation_calls: int = 0
    """Number of runner+judge invocations made during ablation."""

    def top_culprit(self) -> NodeAttribution | None:
        """The highest-confidence culprit, or ``None`` if there are none."""
        return self.culprits[0] if self.culprits else None

    def primary_culprits(self) -> list[NodeAttribution]:
        """All culprits with severity == ``"primary"``."""
        return [c for c in self.culprits if c.severity == "primary"]

    def by_prompt(self) -> list[PromptAttribution]:
        """Roll span-level blame up to the prompt level.

        For each distinct ``prompt_id`` referenced by the culprits,
        aggregates the spans that share it: mean confidence, max
        severity, concatenated reasoning. Culprits with no ``prompt_id``
        are skipped (they have no prompt to optimize).

        This is the view Reflex consumes — it tells you which prompt to
        update and how confident Origin is that the update will help.
        """
        groups: dict[str, list[NodeAttribution]] = {}
        for c in self.culprits:
            if not c.prompt_id:
                continue
            groups.setdefault(c.prompt_id, []).append(c)

        out: list[PromptAttribution] = []
        for prompt_id, spans in groups.items():
            spans_sorted = sorted(spans, key=lambda s: s.confidence, reverse=True)
            mean_conf = sum(s.confidence for s in spans_sorted) / len(spans_sorted)
            max_sev = max(spans_sorted, key=lambda s: _SEVERITY_RANK[s.severity]).severity
            reasoning_parts: list[str] = []
            for s in spans_sorted:
                label = s.node_id or s.node_name
                if s.reasoning:
                    reasoning_parts.append(f"[{label}] {s.reasoning}")
            out.append(
                PromptAttribution(
                    prompt_id=prompt_id,
                    severity=max_sev,
                    confidence=max(0.0, min(1.0, mean_conf)),
                    spans=spans_sorted,
                    reasoning="\n".join(reasoning_parts),
                )
            )
        out.sort(key=lambda p: p.confidence, reverse=True)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "culprits": [c.to_dict() for c in self.culprits],
            "method": self.method,
            "score": self.score,
            "llm_tokens": self.llm_tokens,
            "ablation_calls": self.ablation_calls,
            "raw": self.raw,
        }

    def to_json(self, **kwargs: Any) -> str:
        kwargs.setdefault("ensure_ascii", False)
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Attribution":
        return cls(
            summary=d.get("summary", ""),
            culprits=[NodeAttribution.from_dict(c) for c in d.get("culprits", [])],
            method=d.get("method", ""),
            score=float(d.get("score", 0.0)),
            llm_tokens=int(d.get("llm_tokens", 0)),
            ablation_calls=int(d.get("ablation_calls", 0)),
            raw=dict(d.get("raw", {})),
        )

    def render(self) -> str:
        """Human-readable multi-line rendering, suitable for CLI output."""
        token_parts = []
        if self.llm_tokens:
            token_parts.append(f"llm={_fmt_tokens(self.llm_tokens)}")
        if self.ablation_calls:
            token_parts.append(f"ablation_calls={self.ablation_calls}")
        token_str = f"  tokens={', '.join(token_parts)}" if token_parts else ""
        lines = [
            f"Origin attribution  (method={self.method}, score={self.score:.3f}{token_str})",
            f"  Summary: {self.summary}",
            "",
        ]
        if not self.culprits:
            lines.append("  (no culprits identified)")
            return "\n".join(lines)
        for i, c in enumerate(self.culprits, 1):
            label = c.node_name
            if c.node_id:
                label = f"{c.node_name} (id={c.node_id})"
            lines.append(f"  {i}. {label}  [{c.severity}, confidence={c.confidence:.2f}]")
            lines.append(f"     {c.reasoning}")

        # If there's a prompt-level rollup worth showing, append it.
        prompts = self.by_prompt()
        if prompts:
            lines.append("")
            lines.append("  --- Prompt-level rollup (for Reflex) ---")
            for p in prompts:
                lines.append(
                    f"  prompt={p.prompt_id}  [{p.severity}, confidence={p.confidence:.2f}, spans={len(p.spans)}]"
                )
        return "\n".join(lines)


__all__ = [
    "Attribution",
    "NodeAttribution",
    "PromptAttribution",
    "Severity",
    "VALID_SEVERITIES",
]
