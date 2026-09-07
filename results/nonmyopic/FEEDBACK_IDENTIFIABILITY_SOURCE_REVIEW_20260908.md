# Feedback identifiability and next-source review

Date: 2026-09-08. Targeted literature/source review, not an exhaustive review or
new scientific endpoint. Zero model calls; no paid permission. All old nulls stay
closed. The preceding turn made progress by connecting the structure interface.

## Literature changes the required control

Gupta, Hartford and Liu tested randomized feedback in LLM experimental-design
agents and found little sensitivity in their tested gene-perturbation and molecular
tasks. Classical methods and a prior-guided hybrid were competitive or stronger.
This does not establish that our untested models cannot adapt, but makes mere
history-aware versus no-history comparison insufficient to identify use of the
input-outcome relationship. [Primary paper](https://arxiv.org/html/2509.21403v1).

GOLLuM jointly trains a language encoder and GP using the GP marginal likelihood,
reporting experimental optimization across 23 tasks. Its evidence concerns a
trained probabilistic hybrid, not reliable posterior probabilities from prompting
alone or monotonic h1/h2/h3 predictive BED. The published article is dated
28 August 2026. This supports numerical uncertainty learning as an architectural
option, not permission to transplant its efficacy claim.
[Primary paper](https://www.nature.com/articles/s42256-026-01283-z).

Large Discovery Models couples generative proposals with empirical Bayesian
surrogate acquisition. Its section 3.3 distinguishes full-candidate acquisition
selection from prefix-based beam/MCTS extensions, whose full empirical evaluation
is left to future work. Neither inference-time search depth nor a recurring
optimization loop is automatically our depth of contingent physical experiments.
This is relevant related work for empirical proposal guidance, not proof that our
desired non-myopic claim is already established.
[Primary preprint](https://arxiv.org/html/2608.15669v1).

Our inference: preserve the explicit split between semantic proposal quality,
numeric posterior calibration, and physical lookahead. A known-kernel Gaussian
regression surrogate with fixed targets and squared loss can remove the
answer-dependent covariance that we need for an adaptivity claim. Any proposed
GP alternative must test that opportunity rather than assume better calibration
implies deeper planning gains. See the assumptions/derivation in the existing
NONMYOPIC_RESEARCH_THIRD_PASS_20260908.md; this is not a general objection to GPs.

## Implemented shuffled-feedback control

Added `build_shuffled_feedback_messages` to the new structure-proposal interface.
It takes a caller-specified, prospectively fixed full derangement. Only the pairing
of history inputs and observed values changes. Inputs, value multiset, history
length, public box, priors, noise and model-facing instructions remain identical
to the history-aware arm. No extra model-facing label announces the control.

Validation rejects non-permutations, fixed points, fewer than two observations,
unchanged values, and shuffles that merely reorder outcomes among exchangeable
replicates at identical inputs. It records the explicit assignment and changed-row
count, never retries to obtain a more favourable permutation, and does not mutate
the true numerical fitting history.

This module returns a prompt and audit metadata, not a response or score. It does
not retrofit a fourth arm into any frozen three-arm panel, invent a scientific
pass threshold, or authorize a model call. A future prospective semantic experiment
must budget it explicitly. Useful history dependence requires predictive benefit
over sham feedback, not merely different generated strings.

The existing sealed scorer still expects its original three arms. A future runner
must separately define paired sham evaluation before any responses. It must not
feed sham observations to the numerical fitter: that would confound the proposer
test with deliberately incorrect posterior conditioning. Nor may it drop cases
whose shuffle is uninformative; the prospective panel should be validated first.

## NewtonBench: not a drop-in replacement

Read the pinned AutoSciLab NewtonBench wrapper at source commit
`acf160eb6c96897748dd92b152703b59b74efc05`, blob
`004113d0480777a80d03fb110bfa8c49e2bb6a8c`.
Source path: `autoscilab/oracle/newtonbench.py`.

The wrapper fixes the chosen law version at construction, which is appropriate
for persistent-world sequential experiments. But its printed status, objective
profile and measurement metadata expose difficulty/law version. Any policy adapter
would need an explicit public/private boundary.

Its default equation evaluator executes arbitrary submitted Python, scores
absolute-valued truth/predictions, and excludes NaN pairs. We must not import this
as our secure executable IR or fixed-target loss. Sign errors and invalid targets
need explicit handling, not silent forgiveness. Its evaluation test seed is fixed
at 42; a prospective held-out study needs a separate frozen outcome boundary.

This was wrapper/source inspection only, not a physical-law validity audit or
evidence the benchmark lacks planning opportunity. No NewtonBench outcomes were
generated, no subset was selected on measured gains, and no new benchmark adapter
or paid launch was created. The vendored 12-domain source inventory is available
for a subsequent source-only audit; adoption remains unproved.

## State of the plan

We have not earned a fresh positive sequential result. Banked initial Number Game
beliefs show real conditional adaptivity, but miss the frozen successive-depth
criterion and do not validate the belief model on fresh worlds. Chemistry's tested
opportunity formulations remain weak/null. The new proposal mechanics and sham
control make the next semantic test more identifiable; they do not supply a new
source opportunity or satisfy deliverable C.

Verification: 91 focused tests passed in 2.67 seconds across shuffled feedback,
structure parsing, executable inference, sealed prediction and actual symbolic
search. These are mechanics tests, not evidence of model feedback sensitivity.

Next action must resolve a source/opportunity route before buying proposer calls.
Do not adopt a popular BO benchmark or increase model budget merely to get a
decreasing depth curve. Automation remains paused and the full goal active.
