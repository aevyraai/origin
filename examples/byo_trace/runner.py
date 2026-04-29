# Copyright 2026 Aevyra AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""runner.py — ablation runner and judge for the byo_trace example.

Pass to the CLI with --runner runner.py to enable causal ablation:

    aevyra-origin diagnose trace.json \
      --score 0.1 \
      --rubric rubric.txt \
      --model openrouter/qwen/qwen3-235b-a22b-thinking-2507 \
      --runner runner.py

runner: trace replay — clones the captured trace with one span's output
replaced by a null value. Fast and deterministic for a fully static trace
(no live LLM calls to re-execute).

judge: scores the final reasoning span's output against the rubric:
  1.0  cites the digital subscription policy (pro-rated / first 7 days)
  0.5  neither policy cited
  0.1  cites the physical-product 30-day return policy (wrong document)
  0.0  no reasoning span or non-string output
"""

from __future__ import annotations

from typing import Any

from aevyra_witness import AgentTrace, TraceNode


def runner(original: AgentTrace, overrides: dict[str, Any]) -> AgentTrace:
    """Clone the trace with one span's output replaced."""
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
    return AgentTrace(
        nodes=new_nodes,
        ideal=original.ideal,
        metadata=dict(original.metadata),
    )


def judge(trace: AgentTrace) -> float:
    """Score the final reasoning span's output against the rubric."""
    final = next(
        (n for n in reversed(trace.nodes) if n.kind == "reason"),
        None,
    )
    if final is None or not isinstance(final.output, str):
        return 0.0
    reply = final.output.lower()
    cites_correct_policy = (
        "pro-rated" in reply or "first 7 days" in reply or "digital subscription" in reply
    )
    cites_wrong_policy = "30 days" in reply or "physical product" in reply or "unopened" in reply
    if cites_correct_policy and not cites_wrong_policy:
        return 1.0
    if cites_wrong_policy:
        return 0.1
    return 0.5
