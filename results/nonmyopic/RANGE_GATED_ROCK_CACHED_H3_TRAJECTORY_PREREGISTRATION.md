# Range-Gated Rock Cached-h3 Trajectory Preregistration

Status: frozen after deterministic instrument and independent-audit tests, before
any live late-state smoke response or trajectory response.

## Motivation and Scope

The seed-24186 producer recovered the exact h3 plan on 16/16 proposal cells, but
its frozen exact-identity audit failed on two value-tied control identities and
therefore authorized no trajectory. A separate cross-platform qualification has
since fixed the selector semantics prospectively: values within `1e-12` are
equivalent and canonical order breaks ties. Local and `msc/oat14` artifacts matched
exactly across 16 cells and 96 selector vectors.

This is a new trajectory protocol on fresh model calls and fresh truths. It does
not reinterpret or rerun the failed seed-24186 audit.

## S0: Late-State Serving Gate

- Model: direct-vLLM Gemma 4 26B A4B thinking, 4,096-token reasoning pass and
  512-token bounded forced final.
- Partition: `msc,llm`, excluding `oat12`.
- Fresh model seed: `24187`.
- Interface: successor-grounded fixed K4 roots, JSON-prefix parsing, exact dynamic
  legality, one registered correction attempt.
- Twelve fixed trajectory-prefix cells: start, one and two south moves, then
  representative one-, two-, and three-check `check-5` outcome histories.

S0 passes only if:

1. all 12 cells return four legal fixed-root plans;
2. all 12 accepted cells and every invalid/forced-final event are accounted for;
3. the selected roots on the first three prefixes are exactly
   `move-SOUTH, move-SOUTH, check-5`;
4. at least 75% of all 12 selected roots match exhaustive d3; and
5. scoring makes no model calls.

Any failure stops this line without a prompt, parser, budget, seed, or threshold
repair.

## Conditional S1: Paired Trajectory Confirmation

S1 runs only after S0 passes.

- Fresh seed `24193`, not used by deterministic dry runs.
- 50 paired truth states from the uniform 256-state prior and eight rounds.
- Four arms share each truth:
  - cached Gemma fixed-root h3;
  - identical-root random h3 tails;
  - exhaustive receding d2;
  - exhaustive receding d3 oracle.
- H3 proposal arms operate in rounds 1--6, when at least three actions remain.
  Both use the same exact d2 tail in rounds 7--8.
- Observations use common deterministic uniforms keyed by seed, trial, position,
  checked rock, and repeat count.
- Every plan and control is scored exactly with the registered `1e-12`
  value-equivalence selector.

### Exact Prompt Cache

The provider cache key is the SHA-256 digest of the complete serialized messages.
Only byte-identical prompts can share a response. Every logical decision records
its key, position, history, compiled plans, and hit/miss status. There are exactly
300 logical LLM decisions and at most 64 unique physical proposal calls; exhausting
that hard cap fails closed.

### Frozen Endpoints

Primary paired endpoints:

1. LLM entropy-AUC gain over exhaustive d2.
2. LLM entropy-AUC gain over identical-root random h3.

Corroborating endpoints:

1. truth-log-posterior-AUC gain over exhaustive d2;
2. truth-log-posterior-AUC gain over identical-root random h3;
3. recovery of the exhaustive d3-over-d2 entropy-AUC gain;
4. frequency of taking `move-SOUTH` in both first rounds; and
5. frequency of an on-site check by round three.

The producer gate passes only if all four paired 95% bootstrap lower bounds are
strictly positive, mean exact-d3 recovery is at least 60%, both route frequencies
are at least 75%, every arm is complete and legal, all truths are paired, exactly
300 logical calls are accounted for, no more than 64 physical calls occur, and
rollout scoring makes no model calls.

## Independent Audit

A fresh local replay must independently recompute:

- every cache mapping and physical miss;
- every h3 plan value and stable root selection;
- every matched-random plan;
- every exact d2/d3 action;
- every common-random observation and posterior metric;
- all trajectory aggregates; and
- four fresh 10,000-bootstrap intervals using audit seeds 24189--24192.

The final confirmation passes only if the producer gate and every audit mechanic
pass and the fresh audit intervals satisfy the same four positive-lower-bound,
recovery, and route gates. No failed endpoint may be repaired or replaced.
