# LLM-rooted one-statement support expansion

Zero-call exploratory mechanics, all four completed Luna cases retained. Numerical
code exhaustively substitutes one statement with a source-valid choice having
the same output type; program length, later scope and predecessor requirements
remain intact. Only actually history-compatible roots and children are retained.
No target answers or true programs are loaded. Frozen source grammar unchanged.

| Case | Original behaviors | Expanded syntax | Expanded behaviors | Separating public inputs /32 |
|---|---:|---:|---:|---:|
| 0 | 1 | 9 | 1 | 0 |
| 1 | 7 | 64 | 35 | 32 |
| 2 | 1 | 12 | 2 | 15 |
| 3 | 1 | 93 | 48 | 32 |

History-aware search considered1163 unique candidates using1647 substitution
attempts and1417 filtering evaluations (excluding root eligibility checks).
The blind control has roots only in case2:11 retained programs,11 behaviors,
22 separating inputs. Other blind cases remain unsupported. These are the same
local-search rules, not matched total computation: root counts and lengths differ.

This recovers query-resolvable model uncertainty in two previously concentrated
pools without additional model calls. It does not prove accuracy, Bayesian
calibration, improvement over independent symbolic search, or a horizon gap.
The uniform entropy is a pool diagnostic, not a source-prior posterior estimate.
Finite-panel behavioral distinctions/equivalences must not be generalized to all
inputs. Case0 remains concentrated; no silent deeper mutation or case replacement.

Architectural candidate: LLM compatible roots plus bounded numerical support
expansion, exact restricted-prior weighting, then observation updates and risk
planning. Compare against unexpanded LLM roots and a work-bounded symbolic-only
control on fresh cases before scientific claims. Priorities are predictive
coverage/calibration and an oracle-measured genuine h1/h2/h3 structural gap;
more entropy alone is not the goal. Source syntax weights require explicit
restricted-support interpretation and do not correct selection bias.

One focused test passed in0.62s: source validity, root retention, exactly one edit,
history compatibility, root deduplication and empty incompatible-root handling.
Scoped lint passed. Runtime0.66s for audit command. Account usage unchanged at
220.410962589; no new spend, uncertain$.04 reservation retained. Goal incomplete.
