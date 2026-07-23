# Cleveland Heart Workup 26B Policy Preregistration

Registered 2026-07-23 after the independent exact Cleveland qualification passed and
before any Heart-policy response from Gemma 4 26B was observed.

## Purpose

Test whether a compact non-thinking LLM proposer can express useful two-step,
observation-contingent diagnostic policies in the qualified Heart workup environment.
The LLM proposes only legal branch follow-ups. An exact Bayesian verifier scores the
four complete policies and selects the root; rollout simulation and scoring make no
LLM calls.

## Frozen Interface

- Model: `google/gemma-4-26B-A4B-it`, direct vLLM, thinking disabled.
- Cluster: A100 on `msc,llm`, excluding `oat12`; API cost `$0`.
- Four machine-fixed distinct roots per posterior cell.
- Before workup, root slot zero is always `order:clinical-workup`; the other slots are
  the best remaining one-step bedside queries.
- After workup, roots are the four best remaining one-step queries.
- Every reachable root outcome displays its probability and resulting disease/no-
  disease posterior. Categorical observations have human-readable clinical labels.
- Every branch has a machine-generated indexed menu of all legal second actions.
  Importantly, ordinary-query branches before workup may choose the workup action.
- Response format: four root-keyed integer arrays, one index per listed branch; no
  action names, scores, explanations, or extra fields.
- Temperature `0`, output cap `128`, and one bounded validation retry.

## S0 Serving Gate

- Seed `24136`; ten fixed cells: five before and five after workup, with zero to two
  preceding exact one-step queries.
- Pass only if all ten logical calls resolve, all policies are complete and legal,
  workup roots and branch follow-ups are representable, and usage records zero
  reasoning tokens, zero forced exits, and zero rollout/scoring LLM calls.
- Proposal quality is not an S0 endpoint. If S0 fails, stop this interface revision.

## S1 Proposal Gate

Run only after S0 passes, with fresh seed `24137` and 32 balanced posterior cells:

- 16 unworked cells use one or two seed-hashed legal bedside queries, have distinct
  histories and nonzero target entropy, and are retained only when exhaustive d2
  strictly prefers ordering workup over the exact d1 root.
- 16 worked cells use unused Cleveland rows, follow their exact d2 pre-workup path,
  execute workup, and then execute zero to two exact d1 tests; histories are distinct
  and target entropy remains positive.
- The exact verifier scores each of the four LLM policies over both actions.
- Matched random uses the identical roots and branch menus, sampling one legal
  follow-up independently per branch with a seed derived from the cell index.
- The strong myopic control uses the exact d1 root with its exact optimal second
  action. Exhaustive d2 supplies the opportunity denominator.
- 5,000 paired bootstrap replicates; all cells and controls are computed before any
  endpoint is reported.

S1 passes only if every mechanics gate passes and all four endpoints pass:

1. The 95% paired-bootstrap lower bound for matched-random minus LLM cumulative
   entropy cost is strictly positive over all 32 cells.
2. The lower bound for strong-d1 minus LLM cost is strictly positive over the 16
   unworked d2 opportunities.
3. The exact verifier selects the workup-root policy on at least 75% of unworked
   opportunities.
4. Mean recovery of the exhaustive-d2 improvement over strong d1 is at least 0.60.

A failure stops the Heart LLM-policy line without threshold tuning or a formal
trajectory run. A pass authorizes a separately preregistered paired confirmation run;
it does not itself establish a trajectory-level advantage.
