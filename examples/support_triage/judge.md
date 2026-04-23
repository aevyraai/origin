# Support-triage rubric

Score the triage response 0..1 based on the FULL PIPELINE TRACE.

**1.0 — Correct, grounded, resolves the user's actual problem.**
Acknowledges the duplicate charge, cites the refund policy from
`kb_search`, and confirms the refund has been issued. Tool results
flow through to the reply without contradiction.

**0.6 — Refund is issued but grounding is thin.**
Correct outcome, but the reply doesn't clearly reference the stripe
evidence or the policy. Tool signals are partially wasted.

**0.2 — Denies the refund or reframes the duplicate charge.**
The charges in `stripe_lookup` are identical (same amount, description,
date) and the policy explicitly calls this refund-eligible. Any reply
that concludes "no refund" — or invents an upgrade / promotion / credit
to explain away the duplicate — scores here. The tool data was
available and the agent failed to reason from it.

**0.0 — Empty reply, malformed, or unrelated to the user's question.**

## Grounding rules

- If the agent refused the refund while `stripe_lookup` shows duplicate
  charges, the primary failure is the **planner's eligibility
  decision**, not the responder. Score 0.2.
- If the agent issued a refund but fabricated details (e.g. "your
  refund will arrive in 12 months"), the failure is the **responder**.
  Score 0.6.
- If the tools weren't called at all, the failure is the **planner's
  round-1 tool choice**. Score 0.0–0.2.
