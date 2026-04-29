# Copyright 2026 Aevyra AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Diagnose the coding agent — companion to ``pipeline.py``.

Usage::

    export OPENROUTER_API_KEY=sk-or-...
    python examples/coding_agent/diagnose.py

This script:

1. Runs the coding agent (real Qwen 8B calls) under a Witness tracer.
2. Scores the captured trace with the test-driven judge below.
3. Hands the trace + score + rubric to Origin, which runs critic +
   decomposition + ablation and merges the results.
4. Prints the attribution and the prompt-level rollup Reflex would
   consume.

The judge is *not* an LLM-as-judge — it's a deterministic function that
reads the test result from the final ``run_tests`` span. That's the
right shape for code-generation tasks: you have a hard signal, use it.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
from aevyra_witness import AgentTrace
from aevyra_witness.runtime import trace as witness_trace

from aevyra_origin import Origin
from aevyra_origin.llm import anthropic_llm, openai_llm

from pipeline import DEFAULT_IDEAL, DEFAULT_TASK, coding_agent  # type: ignore[import-not-found]


# ---------------------------------------------------------------------------
# Judge — read the LAST run_tests span and score against pass/fail.
# ---------------------------------------------------------------------------
#
# 1.0  all test cases passed and code compiled
# 0.4  code compiled but one or more test cases failed
# 0.0  code didn't compile, or no run_tests span exists
#
# The judge is intentionally read-only on the trace — it does NOT run the
# code itself. The pipeline already executed the tests; we just consume
# the structured output.


def judge(trace: AgentTrace) -> float:
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


RUBRIC = (
    "Produce a Python implementation that passes every planned test case. "
    "Failures must be attributed to the span responsible — typically the "
    "coder prompt for off-by-one or partition errors, the debugger prompt "
    "for misdiagnosis, or the planner for wrong test cases. Tool spans "
    "(search_docs / check_signature / run_tests) are stubs or harnesses "
    "and should not normally be primary culprits."
)


# ---------------------------------------------------------------------------
# Ablation runner and judge — imported from runner.py so the CLI and
# diagnose.py share the same implementation.
# ---------------------------------------------------------------------------

from runner import judge, runner  # type: ignore[import-not-found]


# ---------------------------------------------------------------------------
# LLM for the attribution methods (NOT for the pipeline — pipeline.py
# uses Qwen 8B; here we want a more capable model to read the trace).
# ---------------------------------------------------------------------------


_ATTRIBUTION_PROVIDER_MAP: dict[str, dict] = {
    "anthropic":  {},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1",  "env_key": "OPENROUTER_API_KEY"},
    "openai":     {},
    "together":   {"base_url": "https://api.together.xyz/v1",   "env_key": "TOGETHER_API_KEY"},
    "groq":       {"base_url": "https://api.groq.com/openai/v1","env_key": "GROQ_API_KEY"},
    "ollama":     {"base_url": "http://localhost:11434/v1",      "api_key": "ollama"},
}


def _pick_llm(model_str: str):
    """Resolve 'provider/model' and return an LLMFn for attribution."""
    parts = model_str.split("/", 1)
    if len(parts) != 2 or parts[0] not in _ATTRIBUTION_PROVIDER_MAP:
        raise SystemExit(
            f"Unknown provider in {model_str!r}. "
            f"Use 'provider/model' format, e.g. 'anthropic/claude-sonnet-4-5'. "
            f"Supported: {', '.join(_ATTRIBUTION_PROVIDER_MAP)}."
        )
    provider, model = parts[0], parts[1]
    cfg = _ATTRIBUTION_PROVIDER_MAP[provider]

    if provider == "anthropic":
        return anthropic_llm(model=model)

    base_url = cfg.get("base_url")
    api_key = cfg.get("api_key")
    if "env_key" in cfg:
        api_key = os.environ.get(cfg["env_key"])
        if not api_key:
            raise SystemExit(f"Provider {provider!r} requires {cfg['env_key']} to be set.")
    else:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("Provider 'openai' requires OPENAI_API_KEY to be set.")
        base_url = os.environ.get("OPENAI_BASE_URL")
    return openai_llm(model=model, base_url=base_url, api_key=api_key)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the coding agent and diagnose failures.")
    parser.add_argument(
        "--model",
        default="openrouter/qwen/qwen3-8b",
        help=(
            "Model for attribution (reading the trace), in 'provider/model' format. "
            "Examples: 'openrouter/qwen/qwen3-8b', 'anthropic/claude-sonnet-4-5', "
            "'openai/gpt-4o'. (default: openrouter/qwen/qwen3-8b)"
        ),
    )
    parser.add_argument(
        "--pipeline-model",
        default="openrouter/meta-llama/llama-3.1-8b-instruct",
        help=(
            "Model for code generation, in 'provider/model' format. "
            "Examples: 'openrouter/meta-llama/llama-3.1-8b-instruct', 'openai/gpt-4o', "
            "'ollama/qwen3:8b'. (default: openrouter/meta-llama/llama-3.1-8b-instruct)"
        ),
    )
    parser.add_argument("task", nargs="?", default=DEFAULT_TASK, help="Coding task to solve.")
    args = parser.parse_args()

    import pipeline as _pipeline_mod
    _pipeline_mod.resolve_model(args.pipeline_model)

    task = args.task
    ideal = DEFAULT_IDEAL

    sys.stderr.write(f"Pipeline model:     {args.pipeline_model}\n")
    sys.stderr.write(f"Attribution model:  {args.model}\n")
    sys.stderr.write(f"Task:               {task}\n\n")

    sys.stderr.write("Step 1/4  Running coding agent (generating and testing code) ...\n")
    with witness_trace(ideal=ideal, metadata={"scenario": "coin_change", "pipeline_model": args.pipeline_model}) as tracer:
        reply = coding_agent(task)
    captured = tracer.finish()

    trace_path = pathlib.Path(__file__).parent / "trace.json"
    trace_path.write_text(json.dumps(captured.to_dict(), indent=2, default=str))
    sys.stderr.write(
        f"          captured {len(captured.nodes)} spans  "
        f"(prompts: {sorted(set(n.prompt_id for n in captured.nodes if n.prompt_id))})\n"
        f"          trace saved → {trace_path}\n\n"
    )

    sys.stderr.write("Step 2/4  Scoring trace ...\n")
    score = judge(captured)
    sys.stderr.write(f"          score = {score:.3f}\n\n")

    if score >= 1.0:
        sys.stderr.write(
            "Code passed all tests — nothing to attribute. Try a harder task:\n"
            "  python diagnose.py 'implement merge sort'\n"
        )
        print(reply)
        sys.exit(0)

    _METHOD_LABELS = {
        "critic": "finding the culprit span",
        "decomposition": "scoring each step against the rubric",
        "ablation": "testing which spans caused the failure",
    }

    def _progress(msg: str) -> None:
        method_key, _, rest = msg.partition(": ")
        if rest == "starting":
            label = _METHOD_LABELS.get(method_key, method_key)
            sys.stderr.write(f"          {label} ...\n")
        elif rest.startswith("done"):
            pass
        elif msg == "merging results ...":
            sys.stderr.write("          combining results ...\n")

    sys.stderr.write("Step 3/4  Running attribution ...\n")
    origin = Origin(llm=_pick_llm(args.model), runner=runner, judge=judge)
    result = origin.diagnose(
        trace=captured,
        score=score,
        rubric=RUBRIC,
        ideal=ideal,
        method="all",
        progress=_progress,
    )
    sys.stderr.write(f"          {len(result.culprits)} culprit span(s)\n\n")

    sys.stderr.write("Step 4/4  Done. Attribution below.\n\n")

    print(result.render())
    print()
    print(f"Score: {result.score:.3f}")
    print(f"Pipeline reply: {reply!r}")

    print()
    print("=== Fix-type breakdown ===")
    for c in result.culprits:
        label = c.node_id or c.node_name
        print(
            f"  [{label}]  severity={c.severity}  fix={c.fix_type}  confidence={c.confidence:.2f}"
        )

    by_prompt = result.by_prompt()
    if by_prompt:
        print()
        print("=== Prompt-level rollup (for Reflex) ===")
        for pa in by_prompt:
            print(
                f"  prompt={pa.prompt_id}  severity={pa.severity}  "
                f"confidence={pa.confidence:.2f}  spans={len(pa.spans)}"
            )
