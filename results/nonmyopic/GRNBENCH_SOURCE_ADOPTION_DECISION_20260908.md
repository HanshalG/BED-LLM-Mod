# GRN source adoption audit: do not adopt unmodified steady-state semantics

## Decision

The preceding goal turn completed the sealed held-out scorer. Before creating
another policy experiment, audit whether the released biological simulator
provides the assumed observation model. This produced new scientific evidence,
not another infrastructure-only increment.

Do not adopt the pinned GRN release unchanged as a converged steady-state
environment. This does not close all GRN research or establish planner failure.
It closes this proposed unmodified interpretation. No altered iteration count,
selected stable subset, LLM call, or policy pilot is authorized by the audit.

## Source and prospective test

The [LLM-AutoSciLab paper](https://arxiv.org/html/2605.24043) describes an active
gene-regulatory discovery benchmark. The pinned
[released implementation](https://github.com/scientific-discovery/LLM-AutoSciLab/blob/acf160eb6c96897748dd92b152703b59b74efc05/autoscilab/oracle/grnbench.py)
provides five motif families and labels returned node activities as steady-state.
Its negative-feedback and toggle routines use exactly 18 and 28 updates,
respectively, with no residual/convergence test.

Freeze `GRNBENCH_STEADY_STATE_SOURCE_AUDIT_PROTOCOL_20260908.json` and runner/tests
at pushed commit `2ab06768` before source evaluations. Source commit is
`acf160eb6c96897748dd92b152703b59b74efc05`, GRN blob
`698079d505d8b5c8bf380e8ecce0872d71eadec6`.

Check all five families x three difficulties x three versions x 33 settings:
all 32 corners of the five published input bounds plus the all-ones baseline.
Every setting is retained. No hidden-world draw or policy is involved.

Compare published output with one more update, requiring all four node states
to agree within `1e-9 + 1e-6 * abs(published_state)`. Separately compare 200
with 201 updates as a numerical diagnostic, not a replacement benchmark.
Only the fixed loop count is changed in an in-memory copy of the pinned
recurrence; all initial states, parameters, update ordering and equations stay
unchanged. No source worktree or original result is edited.

## Complete results

1,485 settings completed in 0.922 seconds, below the frozen 60-second cap.

| Motif | Settings | Published count not fixed | 200 updates not fixed |
|---|---:|---:|---:|
| Activation chain | 297 | 0 | 0 |
| Coherent feedforward | 297 | 0 | 0 |
| Incoherent feedforward | 297 | 0 | 0 |
| Negative feedback | 297 | 56 | 45 |
| Toggle | 297 | 0 | 0 |

Negative-feedback failures occur in easy/medium/hard: 21/18/17 settings.
The largest example is medium/v1 at signal=10 and all perturbation multipliers=4:
reported C changes from 0.0821513 to 3.4807905 after one more update; reporter
scale is 100, so intensity changes from 8.21513 to 348.07905. The same two values
occur at updates 200 and 201. The log1p reporter difference is 3.63445.
This demonstrates a persistent iteration-parity effect in that example, not
small rounding error. Nonconvergence counts alone do not prove every failing
case has the same oscillation mechanism.

Feedforward controls have no recurrent iteration and therefore agree by
construction. Passing the finite corner/baseline grid for the toggle does not
certify its full continuous domain, equilibrium uniqueness or stability.

Result: `grnbench_steady_state_source_audit/20260908-v1/RESULT.json`.
SHA256: `eca3098283a334262856458f8cbfc46378bbfbe802b9b19c9afbc9686a046ba2`.
Five focused synthetic audit tests pass: exact count changes, absence of oracle
construction/imports, full-grid coverage, known oscillation detection, and
source/count/resource failure checks. Scoped lint also passes.

## Other source findings relevant to model proposals

GRN `run` metadata includes true graph edges, motif ID and clean reporter values.
A future observation adapter would need to strip these explicitly. This is an
interface exposure finding, not a claim that a previous LLM consumed those fields.
The released law evaluator also takes absolute predictions and masks invalid
points, which is not the fixed-target failure-preserving scorer we require.

The [Chem universal grammar](https://github.com/scientific-discovery/LLM-AutoSciLab/blob/acf160eb6c96897748dd92b152703b59b74efc05/autoscilab/oracle/chembench.py)
offers a product-inhibition heuristic about proportional suppression across
substrate values. The implemented c2 law instead gives the exact ratio

    r(C_P=0) / r(C_P) = 1 + Km*C_P / (Kp*(Km + C_A)).

Thus relative suppression decreases toward zero as C_A increases; constant
relative inhibition across C_A is not a property of this implementation.
Do not copy heuristic discrimination tips into the new proposer interface
without checking them against the actual model. This analytic source observation
uses no additional oracle measurements and does not establish which historical
prompts, if any, contained the heuristic.

## Implications for the goal

The benchmark remains a deterministic finite-update program and could be studied
as such, but that would be a different scientific interpretation. Treating its
iteration artifacts as biological discovery or planning headroom would be weak.
A separately specified converged-equilibrium or transient-dynamics model would
require a new source/mechanics rationale and validation, not simply more loop
iterations after seeing these failures.

We still do not have a new source-grounded, deployable non-myopic opportunity
that clears the requested successive-depth gains, nor useful LLM proposals
under the new interface. The existing Chemistry pilot, null diagnostics and
Number Game evidence remain unchanged. No paid run or new environment rollout
was launched; cost $0, cluster unused, automation paused. The overall plan and
research goal remain incomplete.
