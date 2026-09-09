# Prospective shared-plan representation qualification

## Question

Does ordinary Python reduce the executable-representation bottleneck enough to
produce useful predictive hypothesis support, compared with the existing DSL?
This is a prerequisite experiment, not a non-myopic efficacy test. Neither a
schema pass nor observed-example fit answers the scientific question.

## Frozen comparison

Six fresh tasks from the pinned RE-ARC source, excluding all 44 IDs in the union
of the latest exact-repair cohort's excluded and selected IDs. Select by the
first six SHA256(`bed-rearc-representation-v1:` + ID) ranks, without replacement,
using metadata only. Freeze the resulting manifest before source execution.
Use demonstration seed 46100, query seeds 46200..46201, and target seeds
46300..46307. Journal all 66 exact source requests once; a failed source gate
closes this cohort without substitution or another seed. No paid call until the
source journal and actual prompt/transport preflight pass.

Each task has one observed demonstration, two public query inputs and eight
public target inputs. Query and target outputs remain sealed. A single Luna
medium call proposes four contrasting verbal mechanisms. Both arms receive the
exact same plan and public history. Each arm makes one eight-program compile
call and one eight-program repair call, retaining all 16 attempted slots.
Only observed-example execution feedback is available. Repair histories are
isolated across arms. Order alternates Python-first and DSL-first by task index.
Seeds are 46400+3*i for plans, +1 for both compilers, +2 for both repairs.
Shared seeds do not imply deterministic provider responses.

Python uses the separately bounded native executor. DSL uses the existing
expression compiler and isolated executor. Invalid batches are rejected
atomically, not partially salvaged or refilled. Each interface supplies its own
executable contract and compiler feedback. Consequently this estimates the
representation-plus-implementation-interface effect, not a pure syntax effect
with identical prompts. Both arms share the same success/mismatch feedback
calculation on the observed grid. No example acquisition, reasoning escalation,
additional repair round, or cross-arm information is part of the intervention.

## Model and cost

Exact `openai/gpt-5.6-luna`, medium reasoning, OpenAI-only provider, no fallback;
reuse the existing named-plan transport contract. Thirty calls total, full
worst-case reservation $0.08 per attempt and $2.40 block ceiling. Existing
131072 prompt-token/16384 completion-token and 65536 request-byte caps apply.
The budget runner must reread authenticated prices/account usage and reserve
before every HTTP attempt. The hard account-wide London-day $5 cap overrides
the block ceiling. No paid execution is authorized by this document alone:
banked source, bounded transport, failure-prefix replay, and immutable bindings
must first be connected and tested. Any size/contract/provider failure terminates
without retry or slot replacement.

## Scoring and gates

Reuse the existing fixed-slot posterior and predictive scoring mathematics.
Uniform mass over unique observation-consistent programs; Python uses canonical
AST identity, DSL graph identity. Neither claims complete semantic deduplication.
Formatting-only duplicates do not multiply prior mass. Failed future executions
retain mass as failure outcomes. Empty support yields unit failure, never an
optimistic reset. Execution-feedback calls use observed examples only.

Before opening labels require Python observed-example coverage on at least four
of six tasks. Otherwise bank initial_coverage_null with all 60 outputs unopened.
After complete forecasts are sealed, open the exact 60 precommitted labels once.
Qualification requires all of:

- Positive Python probability for at least 8 of 12 realized query answers.
- At least three tasks with two distinct positive-mass nonfailure predictions
  on at least one query input.
- Mean whole-grid Brier improvement over DSL at least 0.01 across the 60 outputs,
  with at least two task-level improvements of at least 0.01.
- Python fixed-canvas Brier no worse than DSL.

Report query and target partitions, all per-task losses, validity, observed fit,
failure mass, exact-output coverage, actual tokens/reasoning/cost, and arm order.
Do not treat dependent program slots as independent observations. A tie or
saturated perfect pair is not a positive qualification. No threshold changes
after responses. All pass/null records leave depth_authorized=false.

## Connection to the full goal

A full pass permits designing a separate sequential gate, not claiming success.
That gate must measure outcome-conditioned support refresh, the fidelity of
simulated versus realized updates, and a real same-terminal-objective horizon
opportunity. It must compare paired depths with compute-matched myopic and
random controls on sealed outcomes. Prediction coverage must include the
observations needed for those branches; unsupported observations cannot be
silently omitted. A representation improvement without these properties remains
insufficient for an LLM-native non-myopic claim.
