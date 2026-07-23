# Range-Gated Rock Hierarchical H4 Preregistration

Frozen before any live response from this interface.

## Claim Boundary

This is a new hierarchical LLM-Modulo architecture, not a repair or replacement
of the closed score-free h4 action-sequence lines. The LLM selects semantic rock
targets. A deterministic compiler supplies low-level movement, and an exact
Bayesian verifier selects a root. A passing proposal gate would establish h4
proposal quality only; trajectories require a separately frozen experiment.

## Interface

- Environment: the audited corner-start Range-Gated RockSample[7,8] instance,
  start `(6,6)`, remote accuracy `.55`, on-site accuracy `.95`, horizon four.
- Model: thinking `google/gemma-4-26b-a4b-it`, temperature zero.
- Machine-fixed roots: two geometry-ranked moves and two exact-d1 checks.
- LLM output: four distinct integer rock targets, one per fixed root.
- Information exposed: rock coordinates, posterior `p_good`, each root successor,
  and Manhattan distance from that successor to every rock.
- Information withheld: EIG, plan values, rankings, preferred targets, routes,
  and movement actions.
- Compiler: after each fixed root, take two canonical shortest-path transit slots
  toward its assigned target, then check that target. If already on target, a
  transit slot repeats the target check.
- Exact verifier: scores all four compiled h4 plans; no LLM call occurs in scoring.
- Serving: 4,096 thinking tokens plus a bounded 128-token non-reasoning final
  when needed, one registered correction attempt, JSON-prefix decoding from
  character zero, all physical requests and forced-final events logged.
- Projection: none.

## S0 Serving And Mechanism Smoke

- Fresh seed: `24228`.
- Ten distinct belief cells, up to ten concurrent logical calls.
- Pass only if:
  - all ten cells compile to four legal plans with distinct targets;
  - rock 4 is assigned to the `move-NORTH` root in at least 8/10 cells;
  - the exact `North, North, North, check-4` route is present in at least 8/10;
  - exact h4 selects that route in at least 8/10;
  - exactly ten logical cells are accepted and all usage is accounted.
- A failed S0 stops the architecture without seed, prompt, model, or threshold
  repair. A passed S0 authorizes only S1.

## Conditional S1 Proposal Gate

- Fresh seed: `24229`.
- Sixteen distinct strict h4 opportunity cells.
- Producer bootstrap: 5,000 replicates, seed `24230`.
- Independent audit bootstrap: 5,000 replicates, seed `24231`.
- Paired controls:
  - four random distinct target assignments with the identical compiler and roots;
  - the identical compiled LLM plans truncated to h3;
  - the strongest exact-d3 root closed with exhaustive h4;
  - exhaustive h4 as the opportunity ceiling.
- Pass only if:
  - paired 95% lower bounds are positive against random goals, shared compiled
    h3, and the strongest exact-d3 root;
  - the exact h4 route is selected on at least 75% of cells;
  - mean exact h4 opportunity recovery is at least `.60`;
  - all mechanics, usage, and independent replay checks pass.

The deterministic zero-call reference passes these gates. Under seed `24229`, its
random-goal comparison is `+.423281` with producer interval
`[+.332666,+.484111]`; these values qualify the instrument only and are not model
evidence.

## Spend

- OpenRouter project ceiling: `$110`.
- Spend before this interface: `$40.239433642459844`.
- Remaining before this interface: `$69.76056635754016`.
- S0 run cap: `$0.25`.
- Conditional S1 run cap: `$0.25`.
- No later stage is authorized by this document.
