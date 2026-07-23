# Range-Gated Rock Hierarchical H5 Preregistration

Frozen before any live response from this h5 interface.

## Claim Boundary

The independently audited exact qualification establishes a strict h5-over-h4
opportunity on the focused-prior RockSample[7,8] task. This new experiment asks
only whether a hierarchical LLM-Modulo proposer can recover that opportunity.
The LLM selects semantic rock targets, a deterministic compiler supplies
low-level movement, and an exact Bayesian verifier selects a root.

A passing proposal gate establishes h5 proposal quality only. It is not a
trajectory result, and it does not replace or pool with any h4 experiment.

## Frozen Interface

- Environment: standard RockSample[7,8], start `(6,6)`, remote accuracy `.55`,
  on-site accuracy `.95`, planning horizon five.
- Prior: rock 6 has `p_good=.5`; every other rock has `p_good=.005`.
- Model: thinking `google/gemma-4-26b-a4b-it`, temperature zero.
- Machine-fixed roots: two geometry-ranked moves and two exact-d1 checks.
- LLM output: four distinct integer rock targets, one per fixed root.
- Information exposed: rock coordinates, posterior `p_good`, each root
  successor, and Manhattan distance from that successor to every rock.
- Information withheld: EIG, plan values, rankings, preferred targets, routes,
  and movement actions.
- Compiler: after each fixed root, take three canonical shortest-path transit
  slots toward its assigned target, then check that target. If already on the
  target, a transit slot repeats that target check.
- Exact verifier: scores all four compiled h5 plans; scoring makes no LLM call.
- Serving: 4,096 thinking tokens plus one bounded 128-token non-reasoning final
  when needed, one registered correction attempt, JSON-prefix decoding from
  character zero, and complete request/forced-final accounting.
- Projection: none.

Belief cells vary only observations of secondary rocks. No cell observes the
load-bearing rock 6 before proposal. Every formal S1 cell must independently
recompute as a strict h5 opportunity: exact h4 selects remote `check-6`, exact
h5 selects `move-NORTH`, and the registered route
`North, West, West, West, check-6` attains the exact h5 value.

## S0 Serving And Mechanism Smoke

- Fresh seed: `24239`.
- Ten distinct focused-prior belief cells, up to ten concurrent logical calls.
- Pass only if:
  - all ten cells compile to four legal h5 plans with distinct targets;
  - rock 6 is assigned to fixed `move-NORTH` in at least 8/10 cells;
  - the registered h5 route is present in at least 8/10 cells;
  - exact h5 scoring selects that route in at least 8/10 cells;
  - all exact h5 roots are `move-NORTH`;
  - exactly ten logical cells are accepted and all usage is accounted.

A failed S0 stops this architecture without seed, prompt, model, threshold, or
projection repair. A passed S0 authorizes only the unchanged S1.

## Conditional S1 Proposal Gate

- Fresh seed: `24240`.
- Sixteen distinct strict h5 opportunity cells.
- Producer bootstrap: 5,000 replicates, seed `24241`.
- Independent audit bootstrap: 5,000 replicates, seed `24242`.
- Paired controls:
  - four random distinct target assignments with identical roots and compiler;
  - the identical compiled LLM plans selected using only their h4 prefixes,
    then evaluated as full h5 plans;
  - the strongest exact-h4 root closed with exhaustive exact h5;
  - exhaustive exact h5 as the opportunity ceiling.
- Pass only if:
  - paired 95% lower bounds are positive against random goals, shared compiled
    h4 scoring, and the strongest exact-h4 root;
  - the registered exact h5 route is selected on at least 75% of cells;
  - mean exact h5 opportunity recovery is at least `.60`;
  - every cell remains a strict registered h5 opportunity;
  - all compiler, exact-scoring, usage, and independent replay checks pass.

The deterministic zero-call reference passes all gates. On the frozen S1 cells
its gains are `+.359851` versus random goals (95% CI
`[+.241132,+.450778]`), `+.480620` versus shared h4 scoring
(`[+.478196,+.483850]`), and `+.470080` versus the strongest exact-h4 root.
It selects the registered route 16/16 and recovers `1.0` of the opportunity.
These values qualify the instrument only and are not model evidence.

## Spend

- OpenRouter project ceiling: `$110`.
- Spend before this interface: `$40.29161618245983`.
- Remaining before this interface: `$69.70838381754017`.
- S0 run cap: `$0.25`.
- Conditional S1 run cap: `$0.25`.
- No later paid stage is authorized by this document.
