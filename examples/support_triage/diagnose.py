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

This script is the companion to ``pipeline.py``. It diagnoses *why* the
triage agent gives the wrong refund decision — and tells you what kind of
fix is needed.

The pipeline is a plan-act-respond loop:

    plan (round 1)         — planner dispatches tools
      ├── stripe_lookup    — pull recent charges
      ├── kb_search        — fetch the refund policy
      └── thread_search    — prior support threads
    plan (round 2)         — planner decides eligibility ← the bug lives here
    respond                — responder drafts the reply

The bug: the round-2 planner ignores clear evidence (two identical charges,
an explicit refund policy) and confabulates an upgrade charge. Origin surfaces
``plan (round 2)`` as the primary culprit with ``fix_type="prompt"`` — the
planner prompt needs to anchor the LLM to its tool results rather than letting
it confabulate. This is something Reflex can act on.

If instead the retriever had returned the wrong document, Origin would return
``fix_type="retrieval"`` and you'd fix the index — not the prompt.

Steps:

    1. Instrument + run the pipeline under a Witness tracer
       (``diagnose_pipeline`` handles this).
    2. Score the trace with a user-supplied judge.
    3. Dispatch to Origin's three attribution methods.
    4. Wire up a runner for causal ablation.
    5. Print the attribution with fix_type, plus the prompt-level rollup
       Reflex would consume.

The ablation runner rebuilds a trace with ``overrides[span_id]`` applied
to the matching span's output. In a real pipeline you'd re-run the
agent with that override forced; here the stubbed LLMs make the replay
deterministic without needing a cache.
"""

from __future__ import annotations

import itertools
import os
import sys
import threading
import time
from typing import Any

from aevyra_witness import AgentTrace, TraceNode
from aevyra_witness.runtime import trace as witness_trace

from aevyra_origin import Origin
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


# ---------------------------------------------------------------------------
# LLM for the attribution methods
# ---------------------------------------------------------------------------


_ATTRIBUTION_PROVIDER_MAP: dict[str, dict] = {
    "anthropic": {},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "env_key": "OPENROUTER_API_KEY"},
    "openai": {},
    "together": {"base_url": "https://api.together.xyz/v1", "env_key": "TOGETHER_API_KEY"},
    "groq": {"base_url": "https://api.groq.com/openai/v1", "env_key": "GROQ_API_KEY"},
    "ollama": {"base_url": "http://localhost:11434/v1", "api_key": "ollama"},  # pragma: allowlist secret
}


def _pick_llm(model_str: str):
    """Resolve 'provider/model' and return an LLMFn for attribution."""
    parts = model_str.split("/", 1)
    if len(parts) != 2 or parts[0] not in _ATTRIBUTION_PROVIDER_MAP:
        raise SystemExit(
            f"Unknown provider in {model_str!r}. "
            f"Use 'provider/model' format, e.g. 'openrouter/qwen/qwen3-8b'. "
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
# Progress spinner
# ---------------------------------------------------------------------------


class _Spinner:
    """Print a spinning cursor + message to stderr while work runs in the background."""

    def __init__(self, message: str) -> None:
        self._message = message
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)

    def _spin(self) -> None:
        for ch in itertools.cycle("|/-\\"):
            if self._stop.is_set():
                break
            sys.stderr.write(f"\r{ch}  {self._message} ")
            sys.stderr.flush()
            time.sleep(0.1)
        sys.stderr.write("\r")
        sys.stderr.flush()

    def __enter__(self) -> "_Spinner":
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join()
        sys.stderr.write("\r" + " " * (len(self._message) + 10) + "\r")
        sys.stderr.flush()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the support-triage agent and diagnose failures."
    )
    parser.add_argument(
        "--model",
        default="openrouter/qwen/qwen3-235b-a22b-thinking-2507",
        help=(
            "Model for attribution (reading the trace), in 'provider/model' format. "
            "Examples: 'openrouter/qwen/qwen3-235b-a22b-thinking-2507', "
            "'anthropic/claude-sonnet-4-5', 'openai/gpt-4o'. "
            "(default: openrouter/qwen/qwen3-235b-a22b-thinking-2507)"
        ),
    )
    args = parser.parse_args()

    question = (
        "Hey, I was just charged $29 twice on the same day for my "
        "subscription — I can see both in my bank statement. Can you "
        "issue a refund?"
    )
    ideal = (
        "Acknowledge the duplicate charge, cite the refund policy, "
        "and confirm the refund is being issued."
    )

    llm = _pick_llm(args.model)
    sys.stderr.write(f"Attribution model:  {args.model}\n")
    sys.stderr.write("Pipeline:           deterministic stubs (no LLM calls)\n\n")

    # ------------------------------------------------------------------
    # Step 1 — run the pipeline and capture the trace
    # ------------------------------------------------------------------
    # The pipeline stubs are deterministic (no LLM calls), so this is
    # instant. In a real pipeline this would be your actual agent call.
    sys.stderr.write("Step 1/4  Running pipeline ...\n")
    with witness_trace(ideal=ideal, metadata={"scenario": "duplicate_charge"}) as tracer:
        pipeline_output = triage_agent(question)
    captured_trace = tracer.finish()

    import json as _json
    import pathlib as _pathlib

    _trace_path = _pathlib.Path("trace.json")
    _trace_path.write_text(_json.dumps(captured_trace.to_dict(), indent=2, default=str))

    sys.stderr.write(
        f"          captured {len(captured_trace.nodes)} spans — "
        f"reply: {pipeline_output[:60]!r}\n"
        f"          trace saved → {_trace_path}\n\n"
    )

    # ------------------------------------------------------------------
    # Step 2 — score the trace
    # ------------------------------------------------------------------
    sys.stderr.write("Step 2/4  Scoring trace ...\n")
    score = judge(captured_trace)
    sys.stderr.write(f"          score={score:.3f}\n\n")

    # ------------------------------------------------------------------
    # Step 3 — run attribution (critic + decomposition + ablation)
    # ------------------------------------------------------------------
    # Each method runs exactly once. Origin merges the results internally.
    # Critic and decomposition are one LLM call each.
    # Ablation re-runs the pipeline per candidate span (no LLM needed).
    origin = Origin(llm=llm, runner=runner, judge=judge)

    _current_spinner: list[_Spinner] = []

    _STEP_LABELS = {
        "critic": "Step 3/4  Finding the culprit span",
        "decomposition": "Step 3/4  Scoring each step against the rubric",
        "ablation": "Step 3/4  Testing which spans caused the failure",
    }

    def _on_progress(msg: str) -> None:
        method, _, rest = msg.partition(": ")
        if rest == "starting":
            label = _STEP_LABELS.get(method, f"Step 3/4  {method}")
            sp = _Spinner(f"{label} ...")
            _current_spinner.append(sp)
            sp.__enter__()
        elif rest.startswith("done") and _current_spinner:
            sp = _current_spinner.pop()
            sp.__exit__(None, None, None)
        elif msg == "merging results ...":
            sys.stderr.write("Step 4/4  Merging results ...\n")

    result = origin.diagnose(
        trace=captured_trace,
        score=score,
        rubric=RUBRIC,
        method="all",
        progress=_on_progress,
    )
    sys.stderr.write(f"          {len(result.culprits)} culprit(s)\n\n")

    print(result.render())
    print()
    print(f"Score: {result.score:.3f}")
    print(f"Pipeline reply: {pipeline_output!r}")

    # -----------------------------------------------------------------------
    # Fix-type summary
    # -----------------------------------------------------------------------
    #
    # fix_type tells you where the repair effort belongs:
    #
    #   "prompt"         → the span's prompt needs rewriting (Reflex can help)
    #   "retrieval"      → the retrieval index returned wrong/missing docs
    #   "tool_schema"    → the tool's input schema led the LLM to call it wrong
    #   "routing"        → the pipeline dispatched to the wrong branch/tool
    #   "infrastructure" → timeout, rate limit, auth error, or quota issue
    #   "unknown"        → Origin couldn't determine the fix from the trace
    #
    # In this scenario the planner confabulates despite having correct tool
    # results — so both culprits should be fix_type="prompt".  If kb_search
    # had returned the wrong document you'd see fix_type="retrieval" instead.

    print()
    print("=== Fix-type breakdown ===")
    for c in result.culprits:
        label = c.node_id or c.node_name
        print(
            f"  [{label}]  severity={c.severity}  fix={c.fix_type}  confidence={c.confidence:.2f}"
        )

    prompt_culprits = [c for c in result.culprits if c.fix_type == "prompt"]
    other_culprits = [c for c in result.culprits if c.fix_type != "prompt"]

    print()
    if prompt_culprits:
        print(
            f"  → {len(prompt_culprits)} prompt fix(es) — Reflex can rewrite "
            f"{', '.join(c.prompt_id or c.node_name for c in prompt_culprits if c.prompt_id)}"
        )
    if other_culprits:
        for c in other_culprits:
            print(
                f"  → fix_type={c.fix_type!r} on '{c.node_name}' — prompt rewriting won't help here"
            )

    # -----------------------------------------------------------------------
    # Prompt-level rollup — what Reflex would consume
    # -----------------------------------------------------------------------

    by_prompt = result.by_prompt()
    if by_prompt:
        print()
        print("=== Prompt-level rollup (for Reflex) ===")
        for pa in by_prompt:
            print(
                f"  prompt={pa.prompt_id}  severity={pa.severity}  "
                f"confidence={pa.confidence:.2f}  spans={len(pa.spans)}"
            )
