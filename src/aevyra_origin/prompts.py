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

"""Prompt templates for Origin's attribution methods.

All prompts are kept in this single module so they can be audited,
diffed, versioned, and swapped out wholesale. Each template is a
module-level string with ``{placeholders}`` filled in via
``.format(**kwargs)``.

Both prompts are written for DAG traces: a reasoning step may dispatch
M tools in parallel, and reasoning may span N steps. The LLM is
instructed to cite culprits by span id (``node_id``) when names repeat,
and to reason about where in the DAG the failure originated.

Two templates, one per attribution method:

    CRITIC_PROMPT         LLM-as-critic: reads the full trace and
                          returns ranked culprit spans with reasoning
                          and confidence.

    DECOMPOSITION_PROMPT  Score decomposition: attributes each
                          criterion from the rubric to the span(s)
                          responsible.
"""

from __future__ import annotations


_TRACE_GUIDE = """The trace may be a DAG rather than a simple linear pipeline.
A single reasoning step can dispatch several tool calls in parallel (these
appear as sibling child nodes). The same prompt may fire at many call sites
(e.g. a planner prompt runs once per step), so node NAMES can repeat. Each
span has a unique ``id=...`` shown in its header, and an optional
``Prompt:`` line naming the underlying prompt that produced it.

When you identify a culprit, cite the specific span by both its
``node_name`` (for human readability) AND its ``node_id`` (for
unambiguous reference). If you are not sure which span of a repeated
name is at fault, pick the one whose input/output content most clearly
demonstrates the failure.

Tool-call failures are a first-class concern. When you see a tool span
(kind=tool) — especially an MCP tool call with an ``MCP server:`` line —
distinguish between three failure modes and say which one applies:

  1. WRONG ARGUMENTS — the tool returned a plausible result but was
     called with the wrong input. The reasoning span that dispatched
     this tool is the culprit, not the tool span. The tool did its job.
  2. TOOL ERROR — the tool itself failed (``Error:`` line present, or
     ``error_code`` in metadata). The tool span is the culprit; the
     reasoning that followed may have compounded the failure by not
     handling the error, in which case mark it as contributing.
  3. MISINTERPRETED OUTPUT — the tool returned a correct result but a
     downstream reasoning span read it wrong (e.g. it searched a
     different field, ignored an empty list, picked the wrong entry).
     The downstream reasoning span is the culprit, not the tool span.

Error codes on MCP tool calls (``[code=...]``) usually point at the
tool or server, not the reasoning. Transient codes (e.g. auth, quota,
timeout) are infrastructure issues — mention them but weight them as
contributing rather than as primary prompt-level failures."""


# ---------------------------------------------------------------------------
# LLM-as-critic
# ---------------------------------------------------------------------------

CRITIC_PROMPT = """You are diagnosing why an agent pipeline produced a low-scoring output.
You are given the rubric the judge used, the score the judge assigned, the
reference/ideal output (if any), and the full execution trace — every span
of the pipeline with its input and output.

{trace_guide}

Your task: identify which span(s) in the trace are responsible for the
score. Ground every claim in concrete content from the trace. Do not
speculate about spans or behavior that is not visible in the trace.

===== RUBRIC =====
{rubric}

===== JUDGE SCORE =====
{score}

===== IDEAL OUTPUT =====
{ideal}

===== EXECUTION TRACE =====
{trace_text}

===== INSTRUCTIONS =====
For each culprit span, provide:
- node_id: MUST exactly match an ``id=...`` from the trace above.
- node_name: MUST exactly match the span's name (for human readability).
- severity: "primary" (main cause of the failure), "contributing"
  (partial cause — this span made things worse but is not the root),
  or "minor" (a small issue that slightly degraded the output but was
  not the driver). A trace typically has at most one primary culprit.
- confidence: float in [0.0, 1.0]. How confident you are that this
  span is actually responsible given the evidence in the trace.
- reasoning: one paragraph grounded in the trace. Quote specific
  inputs or outputs when helpful. Do not restate the rubric.

Also provide a one-paragraph ``summary`` giving the overall diagnosis
in plain language — what went wrong, end to end.

Rank culprits by confidence descending. Include only spans that
plausibly contributed; do not pad the list. If the score is actually
fine and no span is at fault, return an empty ``culprits`` array and
say so in the summary.

When the same prompt fires at multiple spans (same name, different
ids), be specific about which span(s) you're blaming — name each one
separately by its id rather than lumping them together.

===== OUTPUT FORMAT =====
Respond with STRICT JSON and nothing else. No prose before or after,
no markdown fences, no commentary. Exactly this shape:

{{
  "summary": "<one paragraph>",
  "culprits": [
    {{
      "node_id": "<exact id from trace>",
      "node_name": "<exact name from trace>",
      "severity": "primary" | "contributing" | "minor",
      "confidence": 0.0-1.0,
      "reasoning": "<one paragraph, grounded in trace content>"
    }}
  ]
}}
"""


# ---------------------------------------------------------------------------
# Score decomposition
# ---------------------------------------------------------------------------

DECOMPOSITION_PROMPT = """You are decomposing an agent pipeline's judge score into per-span
contributions. Given the rubric, the score, the ideal output (if any),
and the full execution trace, break the rubric down into its underlying
criteria and attribute each criterion to the span(s) in the trace that
are responsible for satisfying or failing it.

{trace_guide}

A single criterion may be the responsibility of several spans (e.g. the
classifier set a bad path AND the answer span amplified it). For each
criterion, distribute contribution weights across the responsible spans
so the weights sum to 1.0.

===== RUBRIC =====
{rubric}

===== JUDGE SCORE =====
{score}

===== IDEAL OUTPUT =====
{ideal}

===== EXECUTION TRACE =====
{trace_text}

===== INSTRUCTIONS =====
1. Read the rubric and enumerate the distinct criteria it evaluates.
   These are the high-level requirements the judge is checking against
   (e.g. "the answer uses the correct policy", "the answer is concise",
   "the classifier routes to the right category"). Keep criterion
   descriptions short — one clause each.

2. For each criterion, decide whether the trace satisfied it or not
   based on the judge score and the actual trace content. Use boolean
   ``satisfied``. If partially satisfied, prefer ``false`` and explain
   in reasoning.

3. For each criterion, list the span(s) from the trace whose behavior
   drives it. For each responsible span, assign a ``contribution``
   weight in [0.0, 1.0] representing that span's share of responsibility
   for this specific criterion. Contributions across spans for a single
   criterion must sum to 1.0 (±0.01 tolerance).

4. Both ``node_id`` and ``node_name`` MUST exactly match a span from
   the trace. Do not invent spans. When the same name appears at
   multiple ids, pick the specific id whose content drives the
   criterion — attribute to each span individually, not to the name.

===== OUTPUT FORMAT =====
Respond with STRICT JSON and nothing else. No prose before or after,
no markdown fences, no commentary. Exactly this shape:

{{
  "criteria": [
    {{
      "criterion": "<short description>",
      "satisfied": true | false,
      "nodes": [
        {{
          "node_id": "<exact id from trace>",
          "node_name": "<exact name from trace>",
          "contribution": 0.0-1.0,
          "reasoning": "<one sentence, grounded in trace content>"
        }}
      ]
    }}
  ]
}}
"""


def format_critic_prompt(*, rubric: str, score: str, ideal: str, trace_text: str) -> str:
    """Build a critic prompt with the shared trace guide substituted in."""
    return CRITIC_PROMPT.format(
        trace_guide=_TRACE_GUIDE,
        rubric=rubric,
        score=score,
        ideal=ideal,
        trace_text=trace_text,
    )


def format_decomposition_prompt(
    *, rubric: str, score: str, ideal: str, trace_text: str
) -> str:
    """Build a decomposition prompt with the shared trace guide substituted in."""
    return DECOMPOSITION_PROMPT.format(
        trace_guide=_TRACE_GUIDE,
        rubric=rubric,
        score=score,
        ideal=ideal,
        trace_text=trace_text,
    )


__all__ = [
    "CRITIC_PROMPT",
    "DECOMPOSITION_PROMPT",
    "format_critic_prompt",
    "format_decomposition_prompt",
]
