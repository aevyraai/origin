# Coding-agent rubric

Score the run 0..1 from the FINAL `run_tests` span output.

**1.0 — All tests pass.**
Every planned test case ran to completion and `passed: true`.

**0.4 — Code compiles, but at least one test fails.**
The function body is syntactically valid Python, but at least one
test case returned the wrong value. The agent reached a final answer
but it doesn't satisfy the spec.

**0.0 — Compile error, missing `run_tests` span, or empty results.**
The code didn't even parse / run, or the pipeline never reached the
test stage.

## Grounding rules

- If `run_tests` shows `compile_error`, the failure is in the **coder**
  prompt — the model produced unparseable code. Score 0.0.
- If tests fail on edge cases (k=0 / k=len(arr)-1) but pass on the
  typical case, the failure is usually an **off-by-one** in the coder.
  Score 0.4.
- If the planner's `test_cases` are themselves wrong (e.g., `expected`
  doesn't match what a correct implementation would return), the
  primary culprit is the **planner** — the coder may have written
  correct code that fails the wrong tests.
- If the debugger's diagnosis points away from the real bug and the
  revised code reproduces the same failure, the **debugger** is a
  contributing culprit.
- Tool spans (`search_docs`, `check_signature`, `run_tests`) are
  deterministic stubs / harnesses; they should rarely be primary
  culprits. Treat blame on a tool span as a hint to look at the
  reasoning span that called it.
