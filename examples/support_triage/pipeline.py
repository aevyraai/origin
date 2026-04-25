# Copyright 2026 Aevyra AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Customer-support triage agent used by `tutorial-support-triage.mdx`.

The pipeline is a plan-act-respond loop:

    plan (round 1)         — planner decides which tools to dispatch
      ├── stripe_lookup    (MCP) — pull the user's recent charges
      ├── kb_search        (MCP) — fetch the refund policy
      └── thread_search    (MCP) — prior support threads for this user
    plan (round 2)         — planner decides eligibility from tool results
    respond                — responder drafts the user-facing reply

Both plan spans share ``prompt_id="planner"`` — Reflex updates that
prompt once and every step benefits. The responder carries its own
``prompt_id="responder"``.

The LLM calls are stubbed with deterministic functions so the tutorial
shows a reproducible trace without burning tokens. Swap them for real
provider calls when you run this against your own setup.

Run this file directly to smoke-test the pipeline and print the trace.
"""

from __future__ import annotations

import json
from typing import Any

from aevyra_witness import KIND_REASON, KIND_TOOL
from aevyra_witness.runtime import span, trace


# ---------------------------------------------------------------------------
# MCP tool stubs
# ---------------------------------------------------------------------------
#
# These are deterministic fakes. In a real deployment each would be an
# MCP client call (Stripe, a vector DB, a ticket system, ...). We
# decorate each with `@span` so the runtime records the call shape,
# and tag with ``kind=KIND_TOOL`` so the trace renders cleanly for a
# critic LLM.


@span("stripe_lookup", kind=KIND_TOOL)
def stripe_lookup(customer_id: str) -> list[dict[str, Any]]:
    # Simulated result for the failing scenario: two identical charges.
    return [
        {
            "id": "ch_001",
            "amount": 29.00,
            "currency": "usd",
            "description": "Pro subscription",
            "date": "2026-04-15",
        },
        {
            "id": "ch_002",
            "amount": 29.00,
            "currency": "usd",
            "description": "Pro subscription",
            "date": "2026-04-15",
        },
    ]


@span("kb_search", kind=KIND_TOOL)
def kb_search(query: str) -> dict[str, Any]:
    return {
        "title": "Refund policy: duplicate charges",
        "body": (
            "Duplicate charges posted within 24 hours of one another are "
            "automatically eligible for a refund. Support staff may issue "
            "the refund without further approval."
        ),
    }


@span("thread_search", kind=KIND_TOOL)
def thread_search(customer_id: str) -> list[dict[str, Any]]:
    return []  # no prior complaints


_TOOL_REGISTRY = {
    "stripe_lookup": stripe_lookup,
    "kb_search": kb_search,
    "thread_search": thread_search,
}


# ---------------------------------------------------------------------------
# Planner + responder LLM stubs
# ---------------------------------------------------------------------------


def _planner_decide_tools(user_message: str) -> list[dict[str, Any]]:
    """Round-1 LLM call: planner emits the tool calls it wants run."""
    return [
        {"name": "stripe_lookup", "args": {"customer_id": "cus_42"}},
        {"name": "kb_search", "args": {"query": "duplicate charge refund"}},
        {"name": "thread_search", "args": {"customer_id": "cus_42"}},
    ]


def _planner_decide_eligibility(
    user_message: str,
    tool_results: dict[str, Any],
) -> dict[str, Any]:
    """Round-2 LLM call: planner decides whether the user gets a refund.

    This is the call the tutorial pins the failure on. The stripe data
    shows two identical charges, the policy explicitly makes them
    refund-eligible, and yet the stub confabulates an upgrade. That's
    the bug we want Origin to surface.
    """
    return {
        "diagnosis": (
            "One charge is the monthly Pro subscription renewal; the other "
            "is a prorated upgrade charge posted on the same day."
        ),
        "eligible_for_refund": False,
        "rationale": "Two distinct line items, not a duplicate.",
    }


def _responder_draft(user_message: str, decision: dict[str, Any]) -> str:
    """Final LLM call: the responder turns the decision into a user reply."""
    if decision["eligible_for_refund"]:
        return (
            "Thanks for flagging this — I can see two identical charges on "
            "the same day. I've queued a refund for the duplicate; you'll "
            "see it back on your card within 5 business days."
        )
    return (
        "Thanks for reaching out! I took a look at your account and the two "
        "charges you're seeing are actually separate line items: your "
        "monthly Pro subscription, plus a prorated upgrade charge. Both "
        "are valid, so no refund is needed on our end. Let me know if you "
        "have any questions about the breakdown!"
    )


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------
#
# Using the context-manager form of `span` lets us nest the tool
# dispatch *inside* the plan span so the DAG looks the way we want:
#
#     plan (p1)
#       ├── stripe_lookup
#       ├── kb_search
#       └── thread_search
#     plan (p2)
#     respond
#
# In real pipelines you'd kick off the tool calls concurrently
# (asyncio.gather / ThreadPoolExecutor). The runtime's ContextVars
# propagate across `await` boundaries so the same parenting works for
# async code.


def triage_agent(user_message: str) -> str:
    # --- Plan step 1: decide tools, dispatch them, gather results -------
    with span("plan", kind=KIND_REASON, prompt_id="planner", optimize=True) as p1:
        p1.input = user_message
        tool_calls = _planner_decide_tools(user_message)
        p1.output = {"tool_calls": tool_calls}

        # Tool spans open/close inside p1 — they parent under it automatically.
        tool_results: dict[str, Any] = {}
        for tc in tool_calls:
            fn = _TOOL_REGISTRY[tc["name"]]
            tool_results[tc["name"]] = fn(**tc["args"])

    # --- Plan step 2: decide eligibility from tool results --------------
    with span("plan", kind=KIND_REASON, prompt_id="planner", optimize=True) as p2:
        p2.input = {"user_message": user_message, "tool_results": tool_results}
        decision = _planner_decide_eligibility(user_message, tool_results)
        p2.output = decision

    # --- Respond: draft the user-facing reply ---------------------------
    with span("respond", kind=KIND_REASON, prompt_id="responder") as r:
        r.input = {"user_message": user_message, "decision": decision}
        reply = _responder_draft(user_message, decision)
        r.output = reply

    return reply


# ---------------------------------------------------------------------------
# CLI entrypoint — run under a tracer and print the captured trace
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

    with trace(ideal=ideal, metadata={"scenario": "duplicate_charge"}) as tracer:
        answer = triage_agent(question)
    at = tracer.finish()

    print("=== question ===")
    print(question)
    print()
    print("=== answer ===")
    print(answer)
    print()
    print("=== trace (LLM view) ===")
    print(at.to_trace_text())
    print()
    print(f"=== trace summary: {len(at.nodes)} spans ===")
    for n in at.nodes:
        parent = f" (parent={n.parent_id})" if n.parent_id else ""
        print(f"  [{n.kind}] {n.name}  id={n.id}{parent}")
    print()
    print(f"optimize_prompt_ids = {at.optimize_prompt_ids}")

    # Uncomment to persist the trace for diagnose.py:
    # import pathlib, json as _json
    # pathlib.Path("trace.json").write_text(_json.dumps(at.to_dict(), indent=2, default=str))
