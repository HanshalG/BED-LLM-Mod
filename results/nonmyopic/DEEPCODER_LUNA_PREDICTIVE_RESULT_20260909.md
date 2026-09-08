# Fresh predictive screen: expansion helps on average, coverage remains incomplete

Prospectively frozen d7b40a2f. Four fresh source cases, eight Luna medium calls,
five arms,32 hidden target outputs per case. Complete forecasts were validated
and sealed before target outcomes opened. No prior endpoint was reopened.

| Arm | Mean half-Brier (lower better) | Zero-mass outputs /128 |
|---|---:|---:|
| Luna original compatible pool | .203125 | 26 |
| Luna one-edit expanded pool | .129609375 | 17 |
| History-blind original | 1.000000 | 128 |
| History-blind expanded | 1.000000 | 128 |
| Bounded symbolic search | .281250 | 36 |

All blind pools abstained; penalty1 was prespecified. Symbolic search abstained
on case1; its mean includes that penalty. These comparisons therefore partly
measure support availability, not solely conditional predictive quality. Programs
use exact syntax-prior weights conditional on the retained support; symbolic
search uses uniform compatible expressions. Neither is a full posterior, and
the symbolic comparator is not total-compute matched to LLM plus expansion.

| Case | Luna original | Expanded | Symbolic |
|---|---:|---:|---:|
| 0 | 0 | 0 | 0 |
| 1 | .687500 | .370000 | 1.000000 (abstention) |
| 2 | 0 | 0 | 0 |
| 3 | .125000 | .1484375 | .125000 |

Expansion improves aggregate Brier by36.19%, but wins only one case, loses one,
and ties two. Four cases are not a powered superiority comparison. Zero-mass
targets fall from26 to17, but remain13.28% of128. Aggregate NLL is infinite for
both Luna variants, and cases1/3 retain missing predictive support. Increased
diversity helped one case and diluted useful mass in another. Do not describe
the result as calibrated Bayesian inference or a non-myopic BED win.

Next dependency: support robustness under additional real observations, measured
against independent fresh outcomes and a compute-accounted symbolic control.
Numerical expansion is promising but cannot simply replace missing support with
arbitrary entropy. A validated open-support residual model or productive targeted
regeneration is needed before trusting simulated posterior transitions. Separately,
the previous horizon-null remains unchanged: this modeling result does not supply
the missing structural planning gap. No depth sweep is justified by this screen.

Replay verified all frozen code hashes, regenerated public histories and true
outputs, exact request payloads, source validity, restricted-prior forecasts,
sealed scores and summed raw costs. Tests6/6 passed in.72s before execution,
including carry-reservation, budget race, label sealing and manual Brier fixture.
Forecast SHA50bc04836b05f509054f246aa923e129b3b7f17860d651d61acc36073be6d47c.
Terminal SHA48e5cb3d5dc2c2af3ff6958ef0c392660d9f34ad2ef56c2dd88076a224f19e45.

Eight calls cost$0.02193081, no new uncertain request. Prior interrupted request's
$.04 remains reserved. Live credits/usage/balance245/220.429487979/24.570512021;
posted day spend$.052793985, conservative recorded$.091814205, allowance$4.908185795.
Process exited. Goal unfinished, automation paused. This is exploratory modeling
evidence, not a paper headline or a reopened null.
