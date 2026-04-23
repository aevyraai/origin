# Copyright 2026 Aevyra AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Run Origin on the support-triage pipeline and print the attribution.

Usage:
    python examples/support_triage/diagnose.py

This script is the companion to ``pipeline.py``. It wires everything
together:

    1. Instrument + run the pipeline under a Witness tracer
       (``diagnose_pipeline`` handles this).
    2. Score the trace with a user-supplied judge.
    3. Dispatch to Origin's three attribution methods.
    4. Wire up a runner for causal ablation.
    5. Print the attribution, plus the prompt-level rollup Reflex would
       consume.

The ablation runner rebuilds a trace with ``overrides[span_id]`` applied
to the matching span's output. In a real pipeline you'd re-run the
agent with that override forced; here the stubbed LLMs make the replay
deterministic without needing a cache.
"""

from __future__ import annotations

import os
from typing import Any

from aevyra_witness import AgentTrace, TraceNode

from aevyra_origin import diagnose_pipeline
from aevyra_origin.llm import anthropic_llm, openai_llm

from pipeline import triage_agent  # type: ignore[import-not-found]


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------
#
# A custom judge that grades the run 0–1 from the full trace. For
# brevity we pin the score to the scenario we care about; in practice
# you'd wrap an LLM-as-judge, a ``Verdict`` metric via
# ``judge_from_verdict``, or whatever you already have.


def judge(trace: AgentTrace) -> float:
    """Grade the triage response 0..1 based on the final reply.

    Rubric (for the benefit of readers — not read by the judge):
      1.0  — acknowledges the duplicate charge, cites policy, issues refund
      0.6  — refund issued but grounding is weak
      0.2  — denies refund / invents an upgrade charge  (our failing case)
      0.0  — empty or malformed reply
    """
    respond_node = next((n for n in trace.nodes if n.name == "respond"), None)
    if respond_node is None or not isinstance(respond_node.output, str):
        return 0.0
    reply = respond_node.output.lower()
    if "refund" in reply and ("queued" in reply or "processed" in reply):
        return 1.0
    if "no refund" in reply or "not a duplicate" in reply or "upgrade charge" in reply:
        return 0.2
    return 0.6


RUBRIC = (
    "Acknowledge the duplicate charge, cite the refund policy, and "
    "confirm the refund is being issued. Refusing or reframing the "
    "charges must score low."
)


# ---------------------------------------------------------------------------
# Ablation runner
# ---------------------------------------------------------------------------
#
# A runner replays the pipeline with ``overrides[span_id]`` forcing the
# output of the matching span. For a deterministic stub pipeline like
# this one, the cheapest runner is to clone the captured trace and
# apply overrides in-place — ablating a span downstream of the replaced
# one also matters, but the decomposition + critic methods will catch
# anything ablation misses.


def runner(original: AgentTrace, overrides: dict[str, Any]) -> AgentTrace:
    new_nodes = []
    for n in original.nodes:
        if n.id in overrides:
            new_nodes.append(TraceNode(
                name=n.name,
                input=n.input,
                output=overrides[n.id],
                id=n.id,
                parent_id=n.parent_id,
                kind=n.kind,
                optimize=n.optimize,
                prompt_id=n.prompt_id,
                metadata=dict(n.metadata),
            ))
        else:
            new_nodes.append(n)
    return AgentTrace(nodes=new_nodes, ideal=original.ideal, metadata=dict(original.metadata))


# ---------------------------------------------------------------------------
# LLM for the attribution methods
# ---------------------------------------------------------------------------


def _pick_llm():
    """Pick whichever LLM backend has credentials set."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return anthropic_llm(model="claude-sonnet-4-5")
    if os.environ.get("OPENAI_API_KEY"):
        return openai_llm(model="gpt-4o")
    if os.environ.get("OPENROUTER_API_KEY"):
        return openai_llm(
            model="anthropic/claude-sonnet-4-5",
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
    raise SystemExit(
        "Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or OPENROUTER_API_KEY "
        "before running the diagnose script."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    question = (
        "Hey, I was just charged $29 twice on the same day for my "
        "subscription — I can see both in my bank statement. Can you "
        "issue a refund?"
    )
    ideal = (
        "Acknowledge the duplicate charge, cite the refund policy, "
        "and confirm the refund is being issued."
    )

    result = diagnose_pipeline(
        triage_agent, question,
        judge=judge,
        rubric=RUBRIC,
        llm=_pick_llm(),
        ideal=ideal,
        runner=runner,          # enables ablation under method="all"
        method="all",
        trace_metadata={"scenario": "duplicate_charge"},
    )

    print(result.render())
    print()
    print(f"Score: {result.score:.3f}")
    print(f"Pipeline reply: {result.raw['pipeline_output']!r}")
