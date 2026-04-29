# Copyright 2026 Aevyra AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Runner and judge for the coding agent example.

Pass this file to the CLI to enable ablation:

    aevyra-origin diagnose trace.json \\
      --score <0-1> \\
      --rubric rubric.txt \\
      --model openrouter/qwen/qwen3-8b \\
      --runner runner.py

The runner re-executes the real pipeline with one span's output overridden.
All spans downstream of the override re-run against the new output with real
LLM calls — this is what makes the ablation causally meaningful. Confidence
scores from critic, decomposition, and ablation are then merged.

The judge reads the last run_tests span and scores:
  1.0  all test cases passed
  0.4  compiled but one or more tests failed
  0.0  compile error, or no run_tests span
"""

from __future__ import annotations

import os
import sys
from typing import Any

from aevyra_witness import AgentTrace
from aevyra_witness.runtime import trace as witness_trace

# Allow importing pipeline from the same directory.
sys.path.insert(0, os.path.dirname(__file__))
import pipeline  # type: ignore[import-not-found]
from pipeline import coding_agent  # type: ignore[import-not-found]


def runner(original: AgentTrace, overrides: dict[str, Any]) -> AgentTrace:
    """Re-run the pipeline with one span's output replaced.

    All spans downstream of the override execute for real — the LLM is called
    again with the overridden context. This gives ablation its causal signal:
    if blanking a span changes the score, that span had real impact.
    """
    # Restore the pipeline model from trace metadata so ablation re-runs use
    # the same model that generated the original trace.
    pipeline_model = original.metadata.get(
        "pipeline_model", "openrouter/meta-llama/llama-3.1-8b-instruct"
    )
    pipeline.resolve_model(pipeline_model)

    # Show which span is being blanked.
    total = len(original.nodes)
    blanked_id = next(iter(overrides), None)
    blanked_node = next((n for n in original.nodes if n.id == blanked_id), None)
    span_label = blanked_node.name if blanked_node else "?"
    span_idx = (original.nodes.index(blanked_node) + 1) if blanked_node else "?"
    pipeline.LOG_PREFIX = f"[{span_idx}/{total}] "
    sys.stderr.write(f"\n  ablation {span_idx}/{total}: blanking '{span_label}'\n")

    # Read the task from the original plan span's input.
    plan_span = next(
        (n for n in original.nodes if n.name == "plan" and n.parent_id is None),
        None,
    )
    task = plan_span.input if plan_span and isinstance(plan_span.input, str) else ""

    with witness_trace(
        ideal=original.ideal,
        metadata=dict(original.metadata),
    ) as tracer:
        coding_agent(task, overrides=overrides)

    return tracer.finish()


def judge(trace: AgentTrace) -> float:
    """Score the trace based on the last run_tests span."""
    last_test = next(
        (n for n in reversed(trace.nodes) if n.name == "run_tests"),
        None,
    )
    if last_test is None or not isinstance(last_test.output, dict):
        return 0.0
    out = last_test.output
    if "compile_error" in out:
        return 0.0
    results = out.get("results", [])
    if not results:
        return 0.0
    passed = sum(1 for r in results if r.get("passed"))
    if passed == len(results):
        return 1.0
    return 0.4
