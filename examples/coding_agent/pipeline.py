# Copyright 2026 Aevyra AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Coding agent — a multi-step pipeline that drafts and tests Python code.

This is the second tutorial pipeline. Unlike support_triage (which stubs
the planner / responder LLMs to keep the failure deterministic), this
pipeline calls a *real* small model — Qwen 8B by default — through any
OpenAI-compatible endpoint (OpenRouter, Together, local vLLM, etc.). The
failures you'll see are real model failures, not authored ones.

Pipeline shape::

    plan (planner)                              ── decides algorithm + signature + tests
      ├── search_docs (tool, parallel)          ── stub: returns canned doc snippet
      └── check_signature (tool, parallel)      ── stub: schema-checks the proposed signature
    synthesize (synthesizer)                    ── distills plan + retrieved docs into a recipe
    write_code (coder)                          ── first draft
    run_tests (tool)                            ── exec'd in a subprocess
    [ if tests fail: ]
      diagnose_failure (debugger)               ── reads test output, proposes a fix
      write_code (coder)                        ── revised draft (SAME prompt_id as first)
      run_tests (tool)                          ── re-runs tests
    respond (responder)                         ── final user-facing answer

Span count is 7 on a happy run, 10 when one revision iteration fires.

The two ``write_code`` spans deliberately share ``prompt_id="coder"`` so
``Attribution.by_prompt()`` rolls them up to a single ``coder`` entry —
exactly what Reflex needs to optimize the prompt once and have every call
site benefit. The same is true for the planner / synthesizer / debugger
prompts (each fires once in this pipeline, but share `prompt_id` makes
them future-safe if the agent ever loops further).

Run this file directly to smoke-test the pipeline against your provider
of choice and print the captured trace.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
from typing import Any

from aevyra_witness import KIND_REASON, KIND_TOOL
from aevyra_witness.runtime import span, trace


# ---------------------------------------------------------------------------
# LLM client — provider/model format, same convention as aevyra-origin CLI
#
#   openrouter/qwen/qwen3-8b      (default)
#   openai/gpt-4o
#   ollama/qwen3:8b
#   together/meta-llama/Llama-3-8b-chat-hf
#   groq/llama3-8b-8192
#
# Pin temperature=0 to keep ablation re-runs reproducible.
# ---------------------------------------------------------------------------

_PROVIDER_MAP: dict[str, dict[str, Any]] = {
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "env_key": "OPENROUTER_API_KEY"},
    "openai": {},
    "together": {"base_url": "https://api.together.xyz/v1", "env_key": "TOGETHER_API_KEY"},
    "groq": {"base_url": "https://api.groq.com/openai/v1", "env_key": "GROQ_API_KEY"},
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",  # pragma: allowlist secret
    },
}

# Module-level state set by resolve_model() or argparse in __main__.
MODEL = "qwen/qwen3-8b"
_BASE_URL: str | None = "https://openrouter.ai/api/v1"
_API_KEY: str | None = None


def resolve_model(model_str: str) -> None:
    """Parse 'provider/model' and update module-level MODEL / _BASE_URL / _API_KEY.

    Call this before coding_agent() when the model is chosen at runtime
    (e.g. from CLI args or from diagnose.py).
    """
    global MODEL, _BASE_URL, _API_KEY

    parts = model_str.split("/", 1)
    if len(parts) == 2 and parts[0] in _PROVIDER_MAP:
        provider, bare_model = parts[0], parts[1]
    else:
        raise SystemExit(
            f"Unknown provider in {model_str!r}. "
            f"Use 'provider/model' format, e.g. 'openrouter/qwen/qwen3-8b'. "
            f"Supported providers: {', '.join(_PROVIDER_MAP)}."
        )

    cfg = _PROVIDER_MAP[provider]
    MODEL = bare_model
    _BASE_URL = cfg.get("base_url")
    if "api_key" in cfg:
        _API_KEY = cfg["api_key"]
    elif "env_key" in cfg:
        _API_KEY = os.environ.get(cfg["env_key"])
        if not _API_KEY:
            raise SystemExit(f"Provider {provider!r} requires {cfg['env_key']} to be set.")
    else:
        # openai — use OPENAI_API_KEY, respect OPENAI_BASE_URL for vLLM etc.
        _API_KEY = os.environ.get("OPENAI_API_KEY")
        if not _API_KEY:
            raise SystemExit("Provider 'openai' requires OPENAI_API_KEY to be set.")
        _BASE_URL = os.environ.get("OPENAI_BASE_URL")  # None → OpenAI default


def _client():
    try:
        from openai import OpenAI  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit(
            "openai package not installed — run `pip install aevyra-origin[openai]`."
        ) from exc
    return OpenAI(base_url=_BASE_URL, api_key=_API_KEY)


def _chat(system: str, user: str, *, json_mode: bool = False) -> str:
    """One-shot chat completion. Returns the assistant message content."""
    kwargs: dict[str, Any] = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
    }
    if json_mode:
        # Nudge — many providers honour response_format, others ignore it.
        kwargs["response_format"] = {"type": "json_object"}
    resp = _client().chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


_STEP_LABELS: dict[str, str] = {
    "plan": "model choosing an algorithm and designing test cases",
    "synthesize": "model turning the plan into a step-by-step coding recipe",
    "write_code": "model generating Python code from the recipe (first draft)",
    "diagnose_failure": "model reading the test failures and diagnosing the bug",
    "write_code (revision)": "model rewriting the code based on the diagnosis",
    "respond": "model summarizing whether the function worked and any caveats",
}

# Overridden by runner.py during ablation re-runs to prefix progress lines.
LOG_PREFIX = ""


def _chat_logged(label: str, system: str, user: str, **kwargs: Any) -> str:
    """Like _chat but writes a prefixed progress line to stderr."""
    description = _STEP_LABELS.get(label, label)
    sys.stderr.write(f"  {LOG_PREFIX}{description} ...\n")
    result = _chat(system, user, **kwargs)
    return result


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------
#
# Kept short because Qwen 8B follows long prompts poorly. Each prompt has
# a stable identity (`prompt_id` on the span) so Reflex can target it.


PLANNER_PROMPT = """\
You are a planner for a coding agent. Given a coding task, return STRICT JSON
with these fields and no other commentary:

{
  "algorithm": "<one short phrase naming the algorithm or approach>",
  "signature": "<the Python function signature, e.g. def f(x, y):>",
  "test_cases": [
    {"input": "<expression>", "expected": "<expression>"},
    ...
  ]
}

Include exactly 3 test cases that cover the cases described in the task.
Do not implement the function — only plan.
"""

SYNTHESIZER_PROMPT = """\
You are a synthesizer. Given a plan plus retrieved documentation, distil
a 4-6 sentence implementation recipe that the coder will follow. Highlight
edge cases. No code yet — just the recipe.
"""

CODER_PROMPT = """\
You are a Python coder. Implement the function described in the plan,
following the recipe. Return ONLY the function definition — no markdown
fences, no commentary, no test cases. The first line must be the
`def` statement.
"""

DEBUGGER_PROMPT = """\
You are a debugger. Given the original code and a failing test report,
diagnose the bug in 1-2 sentences and propose a specific fix in plain
English. Do not output code — only a diagnosis the coder will use.
"""

RESPONDER_PROMPT = """\
You are the user-facing responder. Given the final code and the latest
test result, write a 2-3 sentence reply: confirm whether the function
works, summarize what it does, and surface any caveats. Plain prose,
no code in the reply.
"""


# ---------------------------------------------------------------------------
# Stub tools — search_docs, check_signature
# ---------------------------------------------------------------------------
#
# In a real agent these would hit a vector DB and a static type-checker.
# For the tutorial we hard-code plausible responses so the failure has to
# come from somewhere else (the model, not the tools).


def _stub_search(algorithm: str) -> dict[str, Any]:
    return {
        "title": f"{algorithm} reference",
        "snippet": (
            "Quickselect partitions the input around a pivot and recurses into "
            "the side containing the kth element. Standard implementations use "
            "0-indexed k (so k=0 returns the smallest). Lomuto partition is "
            "the most common scheme in tutorials."
        ),
        "url": "https://en.wikipedia.org/wiki/Quickselect",
    }


def _stub_signature_check(signature: str) -> dict[str, Any]:
    # Cheap "static analysis" — just confirm the signature parses.
    try:
        compile(signature + " pass", "<sig>", "exec")
        return {"valid": True, "signature": signature}
    except SyntaxError as exc:
        return {"valid": False, "signature": signature, "error": str(exc)}


# ---------------------------------------------------------------------------
# Real tool — run the produced code against the planned test cases
# ---------------------------------------------------------------------------


def _run_tests(code: str, test_cases: list[dict[str, str]]) -> dict[str, Any]:
    """Execute ``code`` in a subprocess and run each test case.

    Returns a structured result the coder + debugger spans can read.
    """
    harness = textwrap.dedent(
        """
        import json, sys, traceback
        results = []
        try:
            exec(CODE_BLOCK, globals())
        except Exception as exc:
            sys.stdout.write(json.dumps({"compile_error": repr(exc)}))
            sys.exit(0)

        for case in TEST_CASES:
            try:
                got = eval(case["input"], globals())
                want = eval(case["expected"], globals())
                results.append({
                    "input": case["input"], "expected": case["expected"],
                    "got": repr(got), "passed": got == want,
                })
            except Exception as exc:
                results.append({
                    "input": case["input"], "expected": case["expected"],
                    "got": f"<exception: {exc!r}>", "passed": False,
                })
        sys.stdout.write(json.dumps({"results": results}))
        """
    )
    body = f"CODE_BLOCK = {code!r}\nTEST_CASES = {json.dumps(test_cases)}\n" + harness
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(body)
        path = fh.name

    proc = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=10)
    try:
        payload = json.loads(proc.stdout) if proc.stdout.strip() else {}
    except json.JSONDecodeError:
        payload = {"raw_stdout": proc.stdout, "raw_stderr": proc.stderr}

    if "compile_error" in payload:
        return {"passed": False, "compile_error": payload["compile_error"], "results": []}
    results = payload.get("results", [])
    all_passed = bool(results) and all(r["passed"] for r in results)
    return {"passed": all_passed, "results": results}


# ---------------------------------------------------------------------------
# JSON parsing — Qwen sometimes wraps JSON in fences, sometimes not.
# ---------------------------------------------------------------------------


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        # Drop the opening fence + optional language tag, and the trailing fence.
        lines = text.splitlines()
        if lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        # First line is ```python or ```json — drop it.
        return "\n".join(lines[1:]).strip()
    return text


def _parse_plan(raw: str) -> dict[str, Any]:
    """Parse the planner's JSON output and normalise it to the expected schema.

    Qwen sometimes returns a bare string, a list, or a dict without the
    expected keys. We coerce non-dict outputs into an error and drop
    malformed test_case entries so the rest of the pipeline can keep
    going (degraded but not crashed).
    """
    obj = json.loads(_strip_fences(raw))
    if not isinstance(obj, dict):
        raise ValueError(f"planner returned {type(obj).__name__}, expected JSON object")
    test_cases = obj.get("test_cases")
    if not isinstance(test_cases, list):
        test_cases = []
    obj["test_cases"] = [
        tc for tc in test_cases if isinstance(tc, dict) and "input" in tc and "expected" in tc
    ]
    return obj


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------


def coding_agent(task: str, overrides: dict[str, Any] | None = None) -> str:
    """Plan → search/check → synthesize → write → test [→ revise → re-test] → respond.

    ``overrides`` maps span id → forced output. When a span's id appears in
    overrides, its LLM/tool call is skipped and the override value is used
    instead. All downstream spans re-execute normally against the overridden
    output. This is how ablation re-runs work — Origin injects a null output
    for one suspect span and re-runs the rest of the pipeline to see what changes.
    """
    _ov = overrides or {}

    # --- Plan: model decides algorithm + signature + test cases ----------
    with span("plan", kind=KIND_REASON, prompt_id="planner", optimize=True) as p:
        p.input = task
        if p._id in _ov:
            plan = _ov[p._id] if isinstance(_ov[p._id], dict) else {}
            plan.setdefault("test_cases", [])
        else:
            plan_raw = _chat_logged("plan", PLANNER_PROMPT, task, json_mode=True)
            try:
                plan = _parse_plan(plan_raw)
            except (json.JSONDecodeError, ValueError) as exc:
                plan = {"_parse_error": str(exc), "raw": plan_raw, "test_cases": []}
        p.output = plan

        # Tool calls nest inside the plan span via the call stack.
        algorithm = plan.get("algorithm", "<unknown>")
        signature = plan.get("signature", "")
        with span("search_docs", kind=KIND_TOOL) as s:
            s.input = algorithm
            s.output = _ov[s._id] if s._id in _ov else _stub_search(algorithm)
            docs = s.output
        with span("check_signature", kind=KIND_TOOL) as c:
            c.input = signature
            c.output = _ov[c._id] if c._id in _ov else _stub_signature_check(signature)
            sig_check = c.output

    # --- Synthesize an implementation recipe -----------------------------
    with span("synthesize", kind=KIND_REASON, prompt_id="synthesizer", optimize=True) as syn:
        syn.input = {"plan": plan, "docs": docs, "signature_check": sig_check}
        if syn._id in _ov:
            recipe = _ov[syn._id] if isinstance(_ov[syn._id], str) else ""
        else:
            recipe = _chat_logged("synthesize", SYNTHESIZER_PROMPT, json.dumps(syn.input, indent=2))
        syn.output = recipe

    # --- Write code (first draft) ----------------------------------------
    with span("write_code", kind=KIND_REASON, prompt_id="coder", optimize=True) as w:
        w.input = {"task": task, "recipe": recipe, "signature": signature}
        if w._id in _ov:
            code = _ov[w._id] if isinstance(_ov[w._id], str) else ""
        else:
            code = _strip_fences(
                _chat_logged("write_code", CODER_PROMPT, json.dumps(w.input, indent=2))
            )
        w.output = code

    # --- Run tests -------------------------------------------------------
    test_cases = plan.get("test_cases", [])
    with span("run_tests", kind=KIND_TOOL) as t:
        t.input = {"code": code, "test_cases": test_cases}
        t.output = _ov[t._id] if t._id in _ov else _run_tests(code, test_cases)
        test_result = t.output

    # --- Revise if needed (one iteration only) ---------------------------
    if not test_result.get("passed"):
        with span("diagnose_failure", kind=KIND_REASON, prompt_id="debugger", optimize=True) as d:
            d.input = {"code": code, "test_result": test_result}
            if d._id in _ov:
                diagnosis = _ov[d._id] if isinstance(_ov[d._id], str) else ""
            else:
                diagnosis = _chat_logged(
                    "diagnose_failure", DEBUGGER_PROMPT, json.dumps(d.input, indent=2)
                )
            d.output = diagnosis

        with span("write_code", kind=KIND_REASON, prompt_id="coder", optimize=True) as w2:
            w2.input = {
                "task": task,
                "recipe": recipe,
                "signature": signature,
                "previous_code": code,
                "diagnosis": diagnosis,
            }
            if w2._id in _ov:
                code = _ov[w2._id] if isinstance(_ov[w2._id], str) else ""
            else:
                code = _strip_fences(
                    _chat_logged(
                        "write_code (revision)", CODER_PROMPT, json.dumps(w2.input, indent=2)
                    )
                )
            w2.output = code

        with span("run_tests", kind=KIND_TOOL) as t2:
            t2.input = {"code": code, "test_cases": test_cases}
            t2.output = _ov[t2._id] if t2._id in _ov else _run_tests(code, test_cases)
            test_result = t2.output

    # --- Final user-facing reply -----------------------------------------
    with span("respond", kind=KIND_REASON, prompt_id="responder") as r:
        r.input = {"task": task, "code": code, "test_result": test_result}
        if r._id in _ov:
            r.output = _ov[r._id] if isinstance(_ov[r._id], str) else ""
        else:
            r.output = _chat_logged("respond", RESPONDER_PROMPT, json.dumps(r.input, indent=2))

    return r.output


# ---------------------------------------------------------------------------
# CLI — smoke-test
# ---------------------------------------------------------------------------


DEFAULT_TASK = (
    "Write a Python function min_coins(coins, amount) that returns the minimum "
    "number of coins needed to make the given amount using the provided coin "
    "denominations. Return -1 if the amount cannot be made. "
    "For example, min_coins([1, 3, 4], 6) should return 2 (3+3), not 3 (4+1+1). "
    "Include 3 test cases: a case where greedy fails, a case with no solution, "
    "and a case where amount=0."
)
DEFAULT_IDEAL = (
    "A correct dynamic-programming solution that passes all three test cases "
    "(greedy-failure case, no-solution case, amount=0) and a final reply that "
    "confirms the function works."
)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the coding agent pipeline.")
    parser.add_argument(
        "--model",
        default="openrouter/meta-llama/llama-3.1-8b-instruct",
        help=(
            "Model for code generation, in 'provider/model' format. "
            "Examples: 'openrouter/qwen/qwen3-8b', 'openai/gpt-4o', "
            "'ollama/qwen3:8b', 'groq/llama3-8b-8192'. "
            "(default: openrouter/qwen/qwen3-8b)"
        ),
    )
    parser.add_argument("task", nargs="?", default=DEFAULT_TASK, help="Coding task to solve.")
    args = parser.parse_args()

    resolve_model(args.model)
    sys.stderr.write(f"Model: {args.model}\n")
    sys.stderr.write(f"Task:  {args.task}\n\n")
    with trace(
        ideal=DEFAULT_IDEAL, metadata={"scenario": "coin_change", "pipeline_model": args.model}
    ) as tracer:
        answer = coding_agent(args.task)
    at = tracer.finish()

    print("=== task ===")
    print(args.task)
    print()
    print("=== answer ===")
    print(answer)
    print()
    print("=== trace summary ===")
    for n in at.nodes:
        parent = f" (parent={n.parent_id})" if n.parent_id else ""
        print(f"  [{n.kind}] {n.name}  id={n.id}{parent}  prompt={n.prompt_id}")
    print()
    print(f"optimize_prompt_ids = {at.optimize_prompt_ids}")
    print(f"total spans = {len(at.nodes)}")

    # Persist the trace so the CLI can diagnose it:
    #   aevyra-origin diagnose trace.json --score <0-1> --rubric rubric.txt
    import pathlib as _pathlib

    _pathlib.Path("trace.json").write_text(json.dumps(at.to_dict(), indent=2, default=str))
    print("trace saved → trace.json")
