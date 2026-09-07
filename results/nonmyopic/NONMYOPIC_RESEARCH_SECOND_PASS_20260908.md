# Second pass: isolate information value from computation and discovery timing

Follow-up: [third-pass architecture and sequential-policy audit](NONMYOPIC_RESEARCH_THIRD_PASS_20260908.md)
refines the immediate work package and the interpretation of saved Number Game
depths. It preserves the closed-route and no-new-spend boundaries below.

Date: 2026-09-08 (Australia/Melbourne).
Status: planning and retrospective interpretation only. No model calls, new
environment outcomes, experiment authorization, or changes to frozen statuses.
This supersedes the first pass's recommendation priority and primary depth
definition, not its record of sources or results.

## Bottom line

The best next move is NOT a larger ChemBench depth grid. It is a small, auditable
test of genuine receding-horizon planning with an honest shared predictive model,
all feasible actions, and immediate reuse of hypotheses produced by simulation.
ChemBench is the leading candidate, conditional on that test, not the chosen
headline in advance. Number Game remains the banked positive and cheapest place
to audit the objective-versus-horizon distinction.

The first pass was too willing to substitute guaranteed policy improvement for
the requested horizon result, too optimistic about chemistry's demonstrated
opportunity, and did not close the most important computational-discovery loophole.
The strong scientific question is:

> At fixed measurement budget and comparable productive computation, does
> conditioning future experiments on future observations improve prediction in
> an LLM-constructed model space, beyond obtaining better models upfront?

A stronger additional claim is that anticipating evidence-dependent LLM model
generation improves those decisions. That claim needs its own comparison; do not
treat it as established by a positive horizon curve.

## 1. Correct the causal interpretation of the evidence

| Saved evidence | What we can and cannot conclude |
|---|---|
| Number Game pooled128: d3 Brier .1064787 versus myopic EIG .1216563, 12.476% reduction | A useful compound method beats that baseline. Predictive-risk versus EIG acquisition also changes the objective, so the entire improvement cannot be assigned to horizon without a same-loss h1 comparison. |
| Same report: d3 versus cross-fitted d2, 1.817% reduction; paired difference interval [-.00418,+.00030], 44/52/32 wins/ties/losses | The last depth increment remains unresolved and has many ties. The headline gain is not the effect size for powering h3 versus h2. |
| Fresh Qwen source: dynamic versus fixed d3, 2.85% reduction; interval [-.007940,+.001551] | Fresh evidence for the support-discovery contribution was materially weaker than the myopic-EIG comparison. Its registered null remains a null. |
| ChemBench staged corridor: .0734347/.0583445/.0565689 | Successive aggregate gains were 20.55% and 3.04%, but these were policy-ladder levels, with the oracle assumptions below. The frozen failure does not mean zero effect, and the nonzero effect does not establish the requested ordinary horizon result. |
| ChemBench IID sampling: four n=256 replicates had mean normalized selected-action regret .211%, .154%, .072%, .125%, despite median rank correlations .695-.820 | Full-ranking fidelity and decision quality differ. These results do not certify multi-step decisions or real-model calibration, but they undermine a blanket conclusion that numerical action selection was unusable. |

The sampling reference itself was finite, posterior-relative, and one-step. Rare
regret reached 4.80%; error small relative to current risk may still exceed the
small h3-versus-h2 gain. Do not reopen any failed route or reinterpret its gate as
passed. Use these observations only to design a NEW prospective decision-focused
audit on distinct development data.

Local evidence:
- [Pooled Number Game](NUMBER_GAME_CROSSPLANNER_CANONICAL_POOLED128_RESULT.md).
- [Fresh Qwen source](NUMBER_GAME_QWEN_FULLY_FRESH_DAILY_SOURCE_RESULT.md).
- [Staged corridor](CHEMBENCH_STAGED_COMPOUND_CORRIDOR_TERMINAL_RESULT_20260815.md).
- [Sampling fidelity](CHEMBENCH_POSTERIOR_SAMPLING_FIDELITY_RESULT_20260815.md).

## 2. The central loophole: speculative discovery is already computation

Suppose a branch imagines an assay response and asks the LLM for a mechanism that
explains it. If the returned executable mechanism is useful, the real agent has
already computed it. Its availability is not contingent on that imagined response
actually occurring. Only the evidence for it is contingent.

Therefore distinguish:
- A candidate program can be retained now, with its parameters refitted and
  evidence recomputed using ONLY real observations.
- A simulated answer cannot be added to the real likelihood, and fitted branch
  parameters cannot be imported as if they were supported by real data.

This is a first-principles design concern, not a claim that every saved run leaks
observations. But a planner allowed to use a broad imagined model bank while its
terminal predictor is restricted to branch-revealed models can manufacture a
discovery-timing advantage. The old ChemBench split between predictive particles
and represented forecasting models makes this concern concrete.

Required architecture and control:
1. Maintain a public register of every valid executable hypothesis computed so
   far, including counterfactual proposals. Canonicalize and retain provenance.
2. Permit immediate real-history refitting and evidence-based use. Any retention
   or fitting limit must be the same transparent computational limit for all arms.
3. Include an immediate-union control: give same-loss greedy the valid hypotheses
   generated during root planning, score them on real history, then let greedy
   choose. This is a privileged diagnostic isolating selection from generation;
   also run an independently productive matched-budget greedy competitor.
4. Bound the cycle of proposal generation, union, refitting, and replanning before
   the real action. There is no requirement to reach an unbounded fixed point.
5. Never use proposals from the eventual REAL future trajectory in this control.
   Only proposals actually computed before dispatch are eligible.

If immediate reuse removes the advantage, the result is computational model
search, not evidence that anticipating future measurements adds value. That can
still be useful, but it is not our target claim. If the pool would be too large
to fit, charge its processing cost; do not silently give one arm unlimited fits.

This also tempers the word "irreducible": a finite executable problem is not
mathematically inaccessible to classical search. The defensible LLM contribution
is a semantic proposal advantage under measured resource limits, with genuinely
new compositions and meaningful symbolic-search controls.

## 3. Make actual horizon the primary treatment

For shared state s, use terminal predictive risk R(s) and recursion

    V_0(s) = R(s)
    V_h(s) = min_a E[V_(h-1)(T(s, a, Y, U)) | s, a]

Y is a simulated measurement; U is proposal randomness when refresh is enabled.
At each real turn, execute only the first action of the h-step contingent plan,
observe the real outcome, and replan with min(h, remaining measurement budget).
All arms execute the same B real measurements, use the same objective/updater,
and have the same action constraints. A simulated path cannot inspect its hidden
world when constructing policy state or proposals.

Use h=1/2/3 for the primary empirical curve. Under an exact model, greater budget
can reduce optimal risk; that fact does NOT guarantee monotonic terminal loss
across truncated receding-horizon policies at fixed B. Approximation and
misspecification make the latter still less automatic. Prefix planning with a
base-policy tail and incumbent retention is a useful SECONDARY robustness method,
not a replacement for the requested result. A pass only there must be named so.

The old `PolicyLadderPlanner` calls a lower-level policy over the entire remaining
budget. Its level is not this h. Add tests that count actual optimized observation
layers, distinguish full-budget policy-improvement recursion, and check exact tiny
trees independently. Do not try to retrofit old level-labelled results into h.

## 4. Do not prune away the reason to look ahead

`environments/chembench_mopen/costed.py:candidate_actions` selects by immediate gain
and immediate gain/cost. An action with low one-step gain can disappear before
deeper search ever sees it. This does not prove it caused the saved failures;
it does identify a mechanism that a new implementation must exclude.

Analytic sanity example, not a new scientific result: independent fair bits U,V
determine target T=U XOR V. Queries revealing U or V each leave squared-error Bayes
risk at .25, but together reduce it to zero. A once-only noisy T measurement with
75% accuracy immediately reduces risk to .1875. With two queries, greedy takes
the noisy measurement, then one useless-alone bit, finishing at .1875; two-step
planning takes both bits and finishes at zero. An immediate-gain shortlist can
remove both enabling queries. A third lookahead step adds nothing: plateau is
correct here, not an implementation failure.

For a compact candidate menu, search every feasible design at every depth. If
later scaling requires screening, use independent coverage or valid bounds, and
audit discarded actions' multi-step regret. A one-step gain shortlist is not a
safe general-purpose approximation to non-myopic selection.

Similarly, a covariance-only Gaussian risk surrogate may miss nonlinear
complementarity in a multimodal posterior. Begin with full-mixture conditioning
and numerically verified action values, not a moment approximation whose relevance
to this particular gap has not been tested.

## 5. A coherent model of the unknown, not artificial ignorance

Use a single agent-accessible joint predictive law for hypothetical observations
and terminal targets. The terminal predictor must be able to exploit the same
model information as the planner. Held-out truth-family indices are forbidden in
that law; a planner supplied with the true population is only an oracle reference,
not necessarily a rigorous performance bound for its approximate implementation.

If a discrepancy component is necessary, represent uncertainty over functions
or shared latent mechanisms, not independent extra residual noise at each step.
A rollout samples a persistent latent world; action-dependent independent
measurement noise is then drawn conditional on it. A noisy catch-all with no
cross-action structure cannot faithfully predict which diagnostic measurement
will make a later one informative. Conversely, an overly knowledgeable auxiliary
predictor can solve the task without model discovery; test it alone.

For finite candidate support, the numerical posterior is conditional on that
support. Do not claim it is exact Bayes over the entire program space. A proper
global structural prior needs a normalized finite grammar or a proper generative
program prior; penalizing only parameter count need not normalize an unbounded
syntax space. Fix coefficient priors from independent source information, not
data-adapted bounds presented as if they were prior knowledge.

The stochastic proposer must receive the full permitted history and computational
state in both real and simulated updates. A cached atlas is a different transition
model from fresh live generation unless transfer fidelity is established. Matched
randomness couples matching states; it must not make the live agent omniscient
about future API randomness. Test independent proposal banks and real-call transfer
separately. A proposal bank cannot condition on the simulated latent truth itself.

On an exactly enumerable matched-model test, check posterior calibration and
sequential predictive consistency, not just JSON validity. For the M-open
approximation, measure departures rather than pretending exact identities apply.
Keep numerical integration error, predictive-model error, and proposal-transition
error separate. Correlation of predicted gain with one noisy realized entropy
drop is not a complete diagnostic of any of these three.

## 6. Environment choice: contingent chemistry, not a new benchmark search

ChemBench is still the first candidate because executable quantitative hypotheses,
existing source code, and cheap measurements suit the desired division of labour.
But its old oracle gains do not clear the new ordinary-horizon, equal-information
test. Do not invest in a new simulator or full SMC stack before that test.

Use a small source-defined assay menu with raw scalar responses, a tractable
parameter prior, and a finite measurement budget. Four measurements and up to
eight designs are feasibility proposals, not sacred settings. Justify assay
conditions, noise, and mechanism distribution from the source BEFORE outcomes;
do not engineer mandatory unlocks or choose noise to make adjacent gains exceed 5%.
If repeat assays are physically allowed, allow them consistently; they do not
require restarting the old costed-repeat experiment.

Favour genuine mechanism/parameter ambiguity and interventions that resolve it.
Include ordinary low-opportunity worlds rather than only those selected for
oracle disagreement. A new opportunity-enriched benchmark can be legitimate, but
must be labelled as such and cannot establish broad performance on the parent task.

Number Game is the immediate retrospective diagnostic bridge, not a reopened paid
route. ForceBench/BoxingGym is a conditional transfer candidate only after one
clear primary result. A domain swap without evidence for BOTH semantic proposal
quality and ordinary-horizon headroom just restarts the same uncertainty.

## 7. What the updated literature changes

The first pass's 18-source map remains the broad review. This pass rechecked the
closest primary methods and searched more narrowly for noisy active learning,
model discrepancy, and computational discovery. This is not a priority proof.

- Murphy's MDA v4 already combines pool-conditioned executable proposals,
  numerical evidence, model averaging, and task-aware design. It explicitly
  describes finite-pool renormalization without catch-all mass. Its benchmark
  use of a strong proposer is not evidence that our old cheap interface works.
  Our novelty cannot be "LLM proposes, Bayes scores," nor simply replacing EIG
  with predictive risk. [MDA v4](https://arxiv.org/html/2608.09696v4).
- ModelSMC reinforces the distinction between practical implicit LLM proposals
  and idealized probabilistically corrected sampling. Use its limitation to
  label our approximation, not to certify it. [ModelSMC](https://arxiv.org/html/2602.18266).
- Rollout BO work warns that longer lookahead can become counterproductive under
  model error. Our proposal-transition error adds another failure source.
  [Yue and Kontar](https://proceedings.mlr.press/v108/yue20b.html).
- Noisy active-learning work motivates looking beyond ordinary information gain
  when the task is to distinguish decision-relevant equivalence classes. On a
  finite diagnosis-style task, an EC2-style baseline is worth screening; it is not
  a drop-in solution to continuous chemistry. [Golovin, Krause and Ray](https://las.inf.ethz.ch/files/golovin10near.pdf).
- The reported insensitivity of some LLM experimental agents to shuffled outcomes
  supports a direct history-use control, not relying on plausible explanations.
  [Gupta et al.](https://arxiv.org/abs/2509.21403).

The immediate-reuse critique and the proposed experimental ordering are our
reasoning from this project's architecture, not claims attributed to these papers.

## 8. Revised work sequence and explicit decisions

| Step | Deliverable | Decision |
|---|---|---|
| 0: saved-evidence audit | Check whether banked Number Game records identify same-loss h1/h2/h3 and immediate-reuse comparisons; audit semantics first | If an arm was not run or its counterfactual support is absent, say "unidentified"; never infer it from existing trajectories. No new source endpoints. |
| 1: zero-call correctness and runtime | A tiny ordinary-horizon reference planner; known trap and plateau tests; full action coverage; equal-information prediction; immediate-union bookkeeping | Must finish a complete pilot including controls in a bounded working session with durable per-world outputs. Do not rebuild the old long oracle grid. |
| 2: source-only structural opportunity | Ordinary h1/h2/h3 on a NEW public development distribution and a truth-independent predictive model; oracle separately | Need useful residual h3-versus-h2 headroom, not just a strong h3-versus-EIG result. Evaluate discarded-action and immediate-union sensitivities. |
| 3: small new proposer gate | Valid executable proposals, evidence-conditioned predictive benefit, and competitive semantic search under a measured budget | Gate on held-out prediction and consequential decision quality, not exact truth-string recall alone. No full response bank before the small gate. |
| 4: paired end-to-end pilot | Same-loss h1/h2/h3 with real-history refresh; then simulated-refresh and immediate-union variants | Identify objective, horizon, and anticipated discovery effects separately; profile the complete comparison cost. |
| 5: one sealed confirmation | Frozen population, actual h, resources, updater, controls, failures, and world-level power analysis | Open only if pilot effect and variance make the required adjacent-depth AND matched-compute claims feasible. |

Step 2 is a cheap diagnostic, not a prerequisite for another days-long oracle
study. Step 3 then tests whether the agent can represent that opportunity. A
fixed initial LLM ensemble is a useful intermediate instrument; it does not finish
the stronger path-dependent-discovery goal. Full multi-step refresh is introduced
only once ordinary planning works under the actual agent model.

At the pilot, use three main horizon arms plus productive matched-budget h1,
immediate-union h1, real-history-only refresh h3, and random. The main h3 versus
real-history-only h3 contrast shares the real update schedule and isolates
anticipating refresh. Add larger upfront-pool and symbolic-search variants first
on the smaller semantic panel; retain them in confirmation if they challenge the
claimed LLM/discovery mechanism. Keep naive thinking separately labelled rather
than making it the only serious competitor.

Track physical measurements, proposal tokens/cost, model fits, simulation effort,
and wall time separately. A cache hit is not productive matched computation.
Compare performance across common total resource caps as well as raw depth.
Reference uncertainty should be small compared with the proposed adjacent-depth
advantage, not only small compared with total loss. Freeze the precise regret
normalizer, uncertainty procedure, and tail treatment before new audit responses.

Power confirmation from the weaker adjacent contrast and whole-world paired
variation, with proposal randomness nested inside world and families handled at
the appropriate cluster level. Repeated measurement noise is not a substitute for
new worlds. Do not assume 24, 64, or 128 is enough, and do not require every single
world to improve: the claim is population mean improvement with uncertainty and
reported tails. Low-opportunity ties are scientifically expected.

The first pass's $24 stage allocation is a set of speculative ceilings, not a
credible quote for a powered study. Replace it with measured cost per complete
paired world before committing to sample size. The existing $5 London-day limit
and no-cluster instruction remain. This document authorizes no paid execution,
cleanup, closed-route restart, or automation reactivation.

## Expected paper, and honest alternatives

Desired headline: an LLM constructs executable hypotheses, numerical Bayesian
inference updates them, and genuine observation-contingent h1/h2/h3 planning
successively improves held-out prediction at fixed measurement budget while
beating productive myopic computation. Immediate reuse does not erase the effect.
Fresh-family testing and at least one transfer support the broader interpretation.

The strongest extension demonstrates an additional benefit from anticipating
future LLM updates, beyond equally adaptive real-history-only discovery. If that
extension fails, report the narrower non-myopic LLM-model-space result; do not
claim the original strongest goal was achieved. If only a richer upfront pool
helps, write about computational discovery rather than horizon. If only guarded
policy improvement is monotonic, report that distinction. If no fair ordinary
horizon curve survives, stop promoting a positive depth headline.
