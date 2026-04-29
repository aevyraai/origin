# Copyright 2026 Aevyra AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Runner and judge for the support-triage example.

Pass this file to the CLI to enable ablation:

    aevyra-origin diagnose trace.json \\
      --score 0.2 \\
      --rubric rubric.txt \\
      --model openrouter/qwen/qwen3-8b \\
      --runner runner.py

Because the support-triage pipeline uses fully deterministic stubs (no real
LLM calls), trace replay is the right ablation strategy here: clone the
captured trace, replace one span's output with the override value, and
re-judge. No live LLM calls are needed.

This is different from the coding-agent example, where real re-execution is
required because run_tests actually executes code and its output changes
depending on what the coder wrote. For a stubbed pipeline, replay and
re-execution are equivalent — replay is faster and deterministic.

The judge reads the final respond span and scores:
  1.0  refund issued and queued
  0.6  refund mentioned but grounding is weak
  0.2  refund denied or upgrade charge invented
  0.0  empty or malformed reply
"""

from __future__ import annotations

from typing import Any

from aevyra_witness import AgentTrace, TraceNode


def runner(original: AgentTrace, overrides: dict[str, Any]) -> AgentTrace:
    """Trace-replay runner: clone the trace with one span's output overridden.

    For a fully-stubbed pipeline this is equivalent to real re-execution —
    every span's output is deterministic, so there is nothing to re-run.
    """
    new_nodes = []
    for n in original.nodes:
        if n.id in overrides:
            new_nodes.append(
                TraceNode(
                    name=n.name,
                    input=n.input,
                    output=overrides[n.id],
                    id=n.id,
                    parent_id=n.parent_id,
                    kind=n.kind,
                    optimize=n.optimize,
                    prompt_id=n.prompt_id,
                    metadata=dict(n.metadata),
                )
            )
        else:
            new_nodes.append(n)
    return AgentTrace(nodes=new_nodes, ideal=original.ideal, metadata=dict(original.metadata))


def judge(trace: AgentTrace) -> float:
    """Score the trace based on the final respond span."""
    respond_node = next((n for n in trace.nodes if n.name == "respond"), None)
    if respond_node is None or not isinstance(respond_node.output, str):
        return 0.0
    reply = respond_node.output.lower()
    if "refund" in reply and ("queued" in reply or "processed" in reply):
        return 1.0
    if "no refund" in reply or "not a duplicate" in reply or "upgrade charge" in reply:
        return 0.2
    return 0.6
