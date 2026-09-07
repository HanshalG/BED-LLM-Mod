# Initial LLM beliefs support genuine lookahead, with a smaller third-step gain

## Complete, frozen diagnostic

The preceding goal turn implemented the symbolic proposal control. This turn
returns to the project's strongest banked LLM-generated model evidence and asks
whether its INITIAL beliefs support actual observation-contingent horizons.
The protocol, solver and tests were pushed at `38074d94` before computing values.

Use all 32 saved initial supports, seeds28300..28331, from the July cross-fitted
confirmation bank. Stream only the initial hypothesis records and tree seeds.
Do not construct or use future branch supports, regenerated proposals, validation
supports or historical target rules. Isolate the original rule compiler without
loading its adapters or target-expression collection, and replay every saved
initial extension hash and positive-count field.

Each initial pool defines a uniform prior over unique executable rule extensions.
Membership answers are deterministic conditional on a persistent rule. All
integers0..100 are both available queries and the fixed Brier target set, including
queried integers. Every policy has the same three-query budget. This is a new
retrospective model-based estimand, not a reconstruction of an unrun historical
dynamic policy or the old policy-dependent target metric.

All values use exact rational arithmetic. The solver optimizes every contingent
future action within h2/h3. It evaluates h1 over the full three-query deployed
greedy policy. For h2, its first query is selected with a two-step objective;
the remaining two-step policy is then optimal. Equivalent posterior partitions
are collapsed without losing query choices; smallest query breaks exact ties.
When the posterior is resolved, remaining uninformative queries can fill the
budget without changing prediction loss.

## Results

All32 completed in33.968s; maximum1.931s/tree and53,548 cached states, below frozen
30s and250,000-state per-tree limits.

| Full-budget policy | Mean expected Brier | Gain over preceding depth | Better/tied/worse cases |
|---|---:|---:|---|
| h1 | 0.0700803311 | - | - |
| h2 | 0.0644677582 | 8.0088% | 29 / 2 / 1 |
| h3 | 0.0631955085 | 1.9735% | 21 / 11 / 0 |

h3 improves on h1 in31 cases and ties in1, with approximately9.82% aggregate gain.
The h2 regression in one case is retained: ordinary receding horizons do not
provide a universal adjacent-depth improvement guarantee.

The frozen audit required at least5% at BOTH depth increases. Its status is
`initial_opportunity_null` because the h2-to-h3 effect is smaller. Do not relabel
it as a full pass, lower the threshold, select favorable seeds, or import future
hypotheses to enlarge the apparent effect. A null on this conjunction does not
mean there is no useful lookahead effect.

## Mechanism and numerical warning

A separately labelled retrospective check recomputes only two-step root values,
not new h3 trajectories or outcomes. In all21 cases improved by h3, its chosen
first query is STRICTLY worse under the two-step objective than the h2 optimum.
None of those gains arises solely from choosing a different h2-tied root.
This is a concrete short-horizon sacrifice for lower full-budget risk within
the supplied LLM-generated model space.

Independent synthetic tests match the general finite contingent solver. They
also exposed floating tie sensitivity: in one test two rationally equal h2
values rounded differently, changing the eventual three-query outcome. The new
solver resolves this with exact arithmetic; comparison tests implement the same
declared tie rule in the floating reference. This is not a tolerance change to
the scientific gate.12 focused audit/mechanism tests pass in0.70s.

## Interpretation and remaining goal

Unlike the approximate0.5% Chemistry opportunity, these saved initial Number
Game beliefs contain a material first lookahead gain and a smaller additional
third-step gain. This is genuine contingent planning on LLM-generated executable
support, but its risk is averaged under that support itself. It does not prove
calibration or improvement on independent hidden concepts, usefulness over a
productive symbolic proposer, or benefit from anticipating future LLM generation.
The old12.48% dynamic-root result remains a distinct estimand and is unchanged.

The result makes the scale of the missing evidence clearer; it does not authorize
new paid calls, reopen closed Number Game interfaces, or bypass the unfinished
semantic/control/policy/confirmation requirements. The overall goal remains
incomplete. No LLM call, hidden endpoint, cluster job or automation change.

## Artifacts

- `number_game_initial_horizon_audit/20260908-v1/INITIAL_ONLY.json`: exact initial-only projection.
- `.../RESULT.json`: all32 rows, exact fractions, means and original null status.
  SHA256 `fe6e1764a1deb3b0196962bff39457a4fc3888c97ed75956495b7ae6f3171c4a`.
- `.../tree_28300.json` through `tree_28331.json`: per-belief checkpoints.
- `.../MECHANISM.json`: retrospective short-horizon sacrifice analysis; no new gate authority.

Cost this diagnostic: $0. Historical generation costs are not erased or claimed
to have been zero.
