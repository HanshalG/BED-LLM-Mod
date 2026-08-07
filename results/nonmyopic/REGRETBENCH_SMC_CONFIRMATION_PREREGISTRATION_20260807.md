# RegretBench SMC Dynamic Depth-Two Confirmation Preregistration

Date frozen: 2026-08-07, before any RegretBench SMC policy response.

Status: sealed conditional confirmation protocol; no confirmation endpoint is
open.

## Authorization

This confirmation may run only if the sealed Aug 9 SMC development policy is
a literal `passed` result, its independent replay has zero mismatches, its
daily ledger is fully reconciled, and its deterministic development report is
the provisional tier `smc_provisional_development_signal_confirmation_required`.
Any mechanics failure, gated null, partial artifact, failed replay, changed
binding, or previously opened confirmation authorizes nothing.

The confirmation uses the prospectively frozen 64-item `confirmation` split,
whose ordered IDs hash to
`780a0e4e172251be2781729eeb4e591592b996dc9e1cd668b26240e3076660c9`.
It is disjoint from mechanics and development. No task may be filtered,
replaced, reordered, or inspected before execution.

## Parent Bank

The SMC policy consumes eight banked semantic parent slots. The untouched
cohort therefore receives exactly one parent-bank call per task using the
same DeepSeek model, support-recovery prompt, strict schema, temperature, and
parser as development. This stage generates only the eight parent slots and
four questions needed by the SMC policy. It does not run the older dynamic
policy, select an action, score an outcome, or access hidden truth.

The 64 raw roots and minimal controls (`task_id`, first question, provenance)
are frozen before SMC annotation. The parent-bank gate requires exactly 64
accepted requests and attempts, zero retries/provider retries/reasoning/forced
exits, eight raw slots, four distinct questions, passing privacy audits, and
spend within `$0.20`.

## Policy And Endpoints

After the parent bank passes, run the exact frozen development architecture
from `REGRETBENCH_SMC_DYNAMIC_DEPTH2_POLICY_PREREGISTRATION_20260807.md`:

- one fixed-parent reply annotation for each task;
- every root, parent particle, and two-draw conditioned/history-blind SMC
  transition;
- all seven primary policies and identical matched controls;
- root selection frozen before hidden truth access;
- shared realized transitions for policies selecting the same root;
- aligned generated-likelihood terminal Brier as the primary endpoint; and
- the identical mechanics and scientific gates.

The LLM remains load-bearing for semantic particles, reply likelihoods,
retain/revise transitions, and future questions. The hidden CIG remains an
environment-only evaluator. No Luna or other optional baseline is included.
Fresh final regeneration and draw stability remain descriptive and cannot
rescue or reclassify the result.

## Seeds And Counts

- parent bank: `202608405000 + task`;
- SMC annotations: `202608410000 + task`;
- branch CRN: `202608420000 + task*16 + particle*2 + draw`;
- hidden truth: `202608430000 + task`;
- realized first transition: `202608440000 + task`;
- realized final transition: `202608450000 + task`;
- bootstrap: `202608460000`;
- random root: `202608470000 + task`.

The parent bank is exact 64 calls. SMC planning is exact 8,256 calls and
realized execution is at most 512 calls. Total DeepSeek calls are therefore at
most 8,832. The parent-bank cap is `$0.20`, the SMC policy cap is `$3.50`, and
the aggregate confirmation cap is `$3.70` under the account-wide `$5` daily
limit. Calls are never added merely to approach a cap.

## Decision Rule

The confirmation status is determined only after exact independent replay:

- any mechanics failure: no scientific result and no claim;
- mechanics pass but any preregistered science gate fails: confirmed null;
- all mechanics and science gates pass: confirmed SMC result.

The headline gate remains dynamic depth two versus
`smc_myopic_refresh_brier`, with all other frozen gates mandatory. Subgroups,
draw stability, fresh regeneration, task filtering, post-hoc thresholds, and
optional baselines cannot alter the decision.

## Budget-Day Branch

Earliest execution is 2026-08-10 Europe/London. A literal verified Aug 9 SMC
development pass gives this confirmation priority and defers the planned
Bongard Aug 10 spend. If SMC development is null or fails mechanics, this
confirmation stays closed and the existing Bongard schedule is unchanged.
The executor must check live account credits and cumulative usage before the
parent bank, before SMC policy execution, and after each stage.

Freezing this protocol makes zero model calls and costs zero dollars.
