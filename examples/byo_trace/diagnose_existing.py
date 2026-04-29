# Copyright 2026 Aevyra AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Diagnose a trace you already have — no instrumentation, no live agent.

This script demonstrates Origin's *adapter on-ramp*: you bring a trace
captured from your existing observability stack, convert it once with a
30-line adapter, and run Origin against it.

Usage::

    # Default: read sample_langfuse.json
    python diagnose_existing.py

    # Or point at any file with --source
    python diagnose_existing.py --source langfuse --path sample_langfuse.json
    python diagnose_existing.py --source jsonl    --path sample_jsonl.jsonl

The judge here is a tiny string-match function: the canonical right
answer should mention "digital subscription" and "pro-rated", and should
NOT cite the 30-day physical-returns policy. In a real deployment you'd
swap this for a Verdict ``LLMJudge`` via
``aevyra_origin.judges.judge_from_verdict``.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from aevyra_origin import Origin
from aevyra_origin.llm import anthropic_llm, openai_llm

from from_jsonl import from_jsonl_file  # type: ignore[import-not-found]
from from_langfuse import from_langfuse_file  # type: ignore[import-not-found]
from runner import judge, runner  # type: ignore[import-not-found]


RUBRIC = (
    "Answer must cite the digital subscription refund policy specifically — "
    "pro-rated within the first 7 days, no refund after. Citing the "
    "physical-product 30-day return policy is a hard fail (wrong document)."
)


# ---------------------------------------------------------------------------
# LLM for attribution
# ---------------------------------------------------------------------------

_PROVIDER_MAP: dict[str, dict] = {
    "anthropic": {},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "env_key": "OPENROUTER_API_KEY"},
    "openai": {},
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",  # pragma: allowlist secret
    },
}


def _pick_llm(model_str: str):
    """Resolve 'provider/model' and return an LLMFn for attribution."""
    parts = model_str.split("/", 1)
    if len(parts) != 2 or parts[0] not in _PROVIDER_MAP:
        raise SystemExit(
            f"Unknown provider in {model_str!r}. "
            f"Use 'provider/model' format, e.g. 'openrouter/qwen/qwen3-8b'. "
            f"Supported: {', '.join(_PROVIDER_MAP)}."
        )
    provider, model = parts[0], parts[1]
    cfg = _PROVIDER_MAP[provider]

    if provider == "anthropic":
        return anthropic_llm(model=model)

    base_url = cfg.get("base_url")
    api_key = cfg.get("api_key")
    if "env_key" in cfg:
        api_key = os.environ.get(cfg["env_key"])
        if not api_key:
            raise SystemExit(f"Provider {provider!r} requires {cfg['env_key']} to be set.")
    return openai_llm(model=model, base_url=base_url, api_key=api_key)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


_ADAPTERS = {
    "langfuse": from_langfuse_file,
    "jsonl": from_jsonl_file,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=list(_ADAPTERS), default="langfuse")
    parser.add_argument("--path", default=None)
    parser.add_argument(
        "--model",
        default="openrouter/qwen/qwen3-235b-a22b-thinking-2507",
        help=(
            "Model for attribution, in 'provider/model' format. "
            "Examples: 'openrouter/qwen/qwen3-235b-a22b-thinking-2507', "
            "'anthropic/claude-sonnet-4-5', 'openai/gpt-4o'."
        ),
    )
    args = parser.parse_args()

    here = pathlib.Path(__file__).parent
    default_paths = {
        "langfuse": here / "sample_langfuse.json",
        "jsonl": here / "sample_jsonl.jsonl",
    }
    src = pathlib.Path(args.path) if args.path else default_paths[args.source]

    sys.stderr.write(f"Step 1/3  Loading {args.source} trace from {src} ...\n")
    trace = _ADAPTERS[args.source](src)
    sys.stderr.write(
        f"          captured {len(trace.nodes)} spans  "
        f"(prompts: {sorted(set(n.prompt_id for n in trace.nodes if n.prompt_id))})\n\n"
    )

    sys.stderr.write("Step 2/3  Scoring trace ...\n")
    score = judge(trace)
    sys.stderr.write(f"          score = {score:.3f}\n\n")

    sys.stderr.write(f"Attribution model:  {args.model}\n\n")

    sys.stderr.write("Step 3/3  Running attribution (critic + decomposition + ablation) ...\n")
    origin = Origin(llm=_pick_llm(args.model), runner=runner, judge=judge)
    result = origin.diagnose(
        trace=trace,
        score=score,
        rubric=RUBRIC,
        ideal=trace.ideal,
        method="all",
    )
    sys.stderr.write(f"          {len(result.culprits)} culprit span(s)\n\n")

    print(result.render())
    print()
    print(f"Score: {result.score:.3f}")

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

    out_path = here / "result.json"
    out_path.write_text(json.dumps(result.to_dict(), indent=2, default=str))
    sys.stderr.write(f"\n          full result → {out_path}\n")
