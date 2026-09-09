# Why deeper rollout is not the next intervention

The completed qualification is still a gate failure: one improved task, not the
required two. This post hoc audit changes no gate or endpoint authorization.
Previous turn was progress (complete paid evidence); this turn independently
audits the saved initial support, cached predictions and final mixtures.

| Task | Demo0 survivors | Entropy of predicted demo1 answer (nats) | Probability of observed demo1 answer |
| --- | ---: | ---: | ---: |
| bdad9b1f | 3 | 0.636514 | 1/3 |
| 2dee498d | 4 | 0 | 1 |
| 1caeab9d | 4 | 0 | 0 |
| 99b1bc43 | 0 | undefined | undefined |

The important case is 1caeab9d, the sole aware-refresh improvement. All four
demo0-consistent programs predicted the same wrong demo1 output. Their program
entropy was log(4), but their predictive entropy for that input was zero. The
observed answer was outside their predictive support. Exact deterministic
conditioning therefore removed them all. Aware regeneration subsequently found
two programs that generalized to the eight targets.

A simulator sampling answers only from this initial support cannot generate the
observed demo1 branch, at any rollout depth. Therefore it cannot represent the
measured recovery through that branch. This does NOT prove that every action has
zero information, or that every deeper policy would choose the same action:
cached root predictions for demo2 are missing for this task and for two of task0's
three survivors. The audit labels those missing, never as zero information.
No new program execution or outcome was requested to fill the gaps.

Task0 already contains a useful discriminating observation and the final initial
pool generalizes perfectly without fresh model calls. Task1 has four different
programs that agree on both further demonstrations. After three demonstrations,
all surviving final programs agree on each of the eight tested target inputs;
failed pools give the explicit failure outcome. Hypothesis count alone therefore
overstates useful uncertainty here.

The unsolved task's aware proposals have both wrong outputs and invalid outputs,
not a transport crash. Some returned values are not valid grids. The saved repair
still leaves no program consistent across all demonstrations. Its failure cannot
be described solely as lack of thinking tokens or an adapter defect.

## Architecture decision

1. Do not increase depth or run another width sweep of this closed interface.
2. A successor should propose independent semantic mechanism families before
   compiling executable variants; record family and predictive disagreement
   separately from raw program counts. A new request must not simply ask for
   more syntactically different versions of the same explanation.
3. Separate the fixed hypothesis set used to score predictions from proposals
   used to discover new mechanisms. If a planner models discovery, it needs a
   calibrated predictive model with support for plausible observations outside
   its current finite program set, plus a model of the resulting refresh.
   Adding arbitrary epsilon mass to every possible grid is not a usable model of
   which observations can occur or what a later refresh will recover.
4. Qualify that coverage on independent, prospectively frozen source-only
   histories before policy endpoints. Measure observed-answer support, predictive
   scores, output-level diversity, and refresh transition fidelity; use matched
   history-blind and equal-call controls. No true-program injection or selection
   of this one successful task for a new headline.
5. Establish an actual sequential decision opportunity with a fixed total query
   budget and an equal-compute myopic control before attributing gains to depth.
   This qualification only showed additional examples, not query selection or
   a demonstrated non-myopic structural gap.

This is the actionable distinction: Luna showed useful mechanism discovery on
one task, while its current predictive support excluded the answer needed to
trigger that discovery. A credible planning experiment needs to model that
uncertainty, rather than treating discovery as a free consequence of lookahead.

Audit artifact: REARC_SLOT_OPPORTUNITY_AUDIT_20260909.json. Three tests pass,
including guards against process/network dispatch and target-value reads.
No model calls, program executions or new outcomes; cost $0. Source result
SHA b2d610ebb6b1d019d878c175c3e25b34c96077716fa99b8adfd399b998d0eaf7.
Account balance $23.487266431; conservative London Sep9 spend $1.09445509.
Goal remains unachieved; the next work is a prospective semantic-family/predictive
support design and zero-call adversarial checks, not paid endpoint repetition.
