# Path to a substantive depth-improving LLM BED result

Prepared 2026-09-08 (Australia/Melbourne); literature and repository inspection on
2026-09-07 Europe/London. This is a research plan, not a frozen paid protocol or
permission to restart closed experiments. No model calls or new policy outcomes.

## Decision

Pursue a compact, source-grounded ChemBench successor using LLM-proposed executable
models, numerical Bayesian prediction, and terminal-loss contingent planning.
Target a fresh, paired d1 > d2 > d3 predictive-loss result at equal experimental
budget, with a demonstrable LLM contribution and a compute-matched myopic control.
Keep Number Game as banked supporting evidence and finish its existing paper
independently of whether this extension succeeds.

Do not restart the old costed-repeat medium/hard grid. Its execution state is gone,
its full control was computationally impractical, and its oracle assumptions do
not resolve the central LLM uncertainty. Reuse tested components, not its outcome
cohort, gates, or presumed authorization.

The most defensible contribution would be: planning several experiments ahead
improves held-out prediction by anticipating how observations affect both Bayesian
uncertainty and the computational discovery of useful executable models. Strict
successive depth gains must be measured, not enforced by selecting favorable
test outcomes. A plateau is a failure of that stronger claim.

## What the evidence actually supports

The paper claim validator currently passes all 41 bundles, including hashes and
recorded values; the manuscript compiles to six pages. These checks verify
provenance and rendering, not the scientific adequacy of every comparison.

| Evidence | Implication |
|---|---|
| Number Game d3 vs myopic: 12.48% Brier reduction, 95/128 wins, two planners | Useful lookahead already works in an LLM-generated executable concept space. |
| Fresh dynamic-vs-fixed 96: 3.44% Brier reduction, full protocol null from support floors | Dynamic generation may add a smaller benefit; do not promote a partial pass. |
| Fresh answer-conditioned support: 9.51% MSE reduction, benefit correlation .530 | Proposal adaptation can matter; better proposals do not automatically select better queries. |
| Factored ChemBench oracle: .054391/.037817/.031765 MSE | There is an oracle opportunity under that model, not yet a deployable LLM result. |
| ChemBench semantic atlas: 31.35% item compilation, 3.17% truth recall | The old interface does not justify another policy sweep. |
| Continuous inference/planning audits and staged corridor nulls | Neither more particles nor extra stages automatically produce a stable incremental gain. |

Relevant local reports: `CHEMBENCH_FACTORED_MOPEN_ORACLE_RESULT_20260815.md`,
`CHEMBENCH_LLM_PROPOSAL_ATLAS_TERMINAL_RESULT_20260815.md`,
`MDA_EIGHTH_PASS_ATOMIC_DISCOVERY_ARCHITECTURE_20260815.md`, and
`paper/claim_manifest.json`.

## New code findings that change the interpretation

1. `scripts/chembench_factored_mopen_oracle.py:evaluate_primary` passes
   `truth_indices` into `FactoredPolicyLadderPlanner`. The outer runner supplies
   `mixed.truth_indices`. In `environments/chembench_mopen/mechanics.py`,
   `SpeculativePlanner.predictive` uses the likelihood tables for those particles.
   This is intentional oracle population knowledge, not access to the individual
   realized truth, but it must not carry over into an LLM policy experiment.
2. The same planner's `forecast` uses only the represented inference support;
   `leaf_risk` evaluates that forecast under the broader particle distribution.
   Therefore the planner can know more than the predictor is allowed to use.
   Include a full-information-access predictor control. A claimed discovery gain
   that disappears under this control may be a computational handicap effect.
3. `PolicyLadderPlanner.policy_action_value` calls the FULL remaining-budget
   `policy_value` at level minus one. Consequently d2/d3 here are recursive policy
   improvement levels, not ordinary two-/three-observation truncated horizons.
   This explains both the improvement property and unexpectedly expensive trees.
   Preserve this distinction in new experiments and old-result descriptions.
4. A d1 replay that hits a cache built by d3 verifies reuse, but does not alone show
   that a myopic competitor used the same resources productively. Add a wider,
   better-estimated myopic competitor with the same total resource allowance.

These are scope and design concerns, not evidence that recorded oracle arithmetic
or saved Number Game results are incorrect.

## First-principles specification

### One objective, one experimental budget

Choose an external task loss and a fixed total experiment budget B. For chemistry,
use mean squared log-rate prediction error on a sealed intervention distribution;
report predictive log score and structural recovery secondarily. All depths get
the same initial observations, admissible experiments, and budget. An episode is
a sequence of real measurements, not a longer chain-of-thought prompt.

Write J(pi) = E[L(g(H_B), Z)] for the expected terminal loss of the deployed
predictor g after executing policy pi. The expectation includes environment noise
and stochastic hypothesis proposals. Code evaluates the loss. The LLM does not
assign entropy, confidence weights, likelihood values, or experiment utilities.
Task loss is standard decision-theoretic BED; see Action-BED [3].

### What can be monotonic

For a fixed predictive model and horizon B, define nested classes C_d of complete
contingent policies. C_d optimizes the first d observation-conditioned decisions
and follows the SAME adaptive base policy for the remaining budget. Because the
next optimized decision can copy the base action, C_d is a subset of C_(d+1).
Exact minimization then gives min(C_(d+1)) J <= min(C_d) J. This is a simple
policy-class inclusion argument, not a new theorem or a guarantee of strict gain.

Execute the selected complete contingent policy. Unrestricted rolling replanning
changes the policy and needs its own policy-improvement argument. In the practical
search, explicitly retain the incumbent complete policy, evaluate alternatives
over the same full budget, and accept a replacement only on independent simulated
validation. Report actual maximum optimized contingent depth and search effort.
Label recursive rollout-improvement iterations separately, rather than calling
them ordinary horizon depth. Bertsekas [5] and Yue/Kontar [6] provide the relevant
rollout foundations and model-error cautions.

Naming matters: in this construction even a one-decision optimized prefix includes
the base policy's future cost, so it is already non-myopic. Report the unmodified
greedy policy separately as the baseline, label these curves by optimized prefix
depth, and ALSO report conventional receding-horizon h=1/2/3 as an unguarded
ablation. Do not relabel improvement iterations or a base-policy tail as additional
explicit contingent depth. If only guarded prefix planning improves, that is the
precise claim; ordinary receding-horizon monotonicity remains unestablished.

If every compared policy has value error at most epsilon under the real target
distribution, an estimated improvement greater than 2*epsilon certifies a real
improvement. This elementary bound exposes the key problem: simulation CIs bound
sampling noise, not model bias. We do NOT have a uniform real-world epsilon.
Therefore the guarantee belongs to the stipulated model; strict held-out
improvement belongs to a separate empirical result with simultaneous uncertainty.
Never use cumulative-minimum test curves, isotonic smoothing, or test-driven
fallbacks to manufacture monotonicity.

### A genuine reason for deeper experiments

Look for complementary measurements: one observation changes which later
measurement is useful. For example, a substrate/inhibitor contrast can distinguish
whether a second titration should resolve affinity or maximum-rate effects; a
further condition can separate a remaining interaction. This is a hypothesis
about source kinetics, not a prewritten successful route. All measurements remain
available initially. Parameter uncertainty and observation noise must prevent
immediate saturation without being selected to favor our algorithm.

Measure this opportunity on development worlds. Include both natural low-gap and
high-gap families, with a declared sampling rule; report the full distribution.
Adaptive submodularity [7] explains why some problems favor greedy strategies;
violating it is neither necessary nor sufficient for our desired strict gains.

### Bayesian support and proposal coherence

The LLM is a computational proposal mechanism q(m|history), not automatically a
prior or posterior. Define a public, history-independent structural prior p0(m)
and parameter priors. Canonicalize programs; do not reward repeated strings with
extra probability mass. Recompute likelihoods of newly found models on the full
history. Evidence is parameter-integrated likelihood, not best-fit residual.

For an explicitly enumerated candidate subset, normalize p0(m)*p(history|m) and
call it the posterior conditional on that subset. This is not the full posterior
over undiscovered programs. If using importance particles, account for proposal
densities or state the approximation; likelihood-only weighting of history-based
LLM proposals does not establish exact SMC. ModelSMC explicitly discusses this
limitation [9]. Test order/duplicate sensitivity and calibration on a tiny grammar
where the exact posterior is available.

An M-open planner also needs a distribution for outcomes its current models miss.
It cannot anticipate discovery while giving those outcomes zero probability.
Use a source-trained, calibrated discrepancy/auxiliary predictive component if
needed, with uncertainty learned before confirmation. Give every policy AND the
terminal predictor access to it. Do not insert held-out mechanism lists, true
parameter settings, or evaluator-only forecasts into planning. Audit whether the
auxiliary predictor already solves the task without LLM model creation.

## Literature map and implications

Broad web searches covered LLM BED, sequential design, rollout improvement,
adaptive submodularity, mechanistic discovery, model misspecification, and recent
2026 discovery systems. Primary full texts were inspected for the most relevant
methods; some peripheral papers were screened at abstract level. This is a broad
targeted review, not an exhaustive systematic review or a priority guarantee.

| Source | What it changes for us |
|---|---|
| [1] BED-LLM (2025/26) | Retention, history consistency, and candidate ranking are useful machinery. It does not establish monotonic non-myopic gains. |
| [2] CA-BED (2026) | Conversation trees already exist. Calibration and stopping choices affect its comparisons; tree search alone is not our novelty. |
| [3] Action-BED (June 2026) | Direct expected task loss is a principled objective and can avoid nested information estimation. |
| [4] DAD (2021) | Amortized sequential policies and total-information bounds are established; our novelty must involve LLM model construction. |
| [5] Bertsekas rollout review (2022) | Use base-policy completion and explicitly distinguish improvement iterations from truncated horizon. |
| [6] Yue/Kontar (2020) | Longer lookahead can amplify misspecification; a useful horizon depends on model quality. |
| [7] Golovin/Krause (2011, corrected version 2017) | Measure experimental complementarity rather than assuming greedy leaves headroom. |
| [8] MDA v4 (Aug 25, 2026) | Use executable proposals and numerical inference. The revision reports prompt-leak repairs and adds BoxingGym; old prompt assumptions need re-audit. |
| [9] ModelSMC v2 (June 2026) | Context-dependent proposal weights have an explicit practical-versus-idealized distinction. |
| [10] LLM-AutoSciLab (May 2026) | Supplies ActiveSciBench-Chem/GRN and a relevant closed-loop model-discovery baseline. |
| [11] LLM-ACES (June 2026) | Operator priors and adaptive equation search motivate a strong non-LLM-search comparison. |
| [12] BoxingGym (2025) | Independent generative environments and numerical prediction are useful transfer targets. |
| [13] Hypothesis generation/updating (May 2026) | Number Game probes disagree; thinking changes inference biases rather than guaranteeing calibration. |
| [14] LLMs for BO: Are We There Yet? (2025) | Shuffled-outcome controls can expose apparent adaptation driven by prior knowledge alone. |
| [15] Large Discovery Models v2 (Aug 30, 2026) | Generators plus uncertainty-aware surrogates are now a crowded design space; the sequential contingent contribution needs isolation. |
| [16] GOLLuM (Aug 28, 2026) | LLM representations plus calibrated numerical models are another strong hybrid route, not proof of depth gains. |
| [17] Robust design via generalized Bayes (2025) | Noise/model robustness warrants tests; changing the inferential target must be acknowledged. |
| [18] BED via score matching (July 2026) | Amortization is relevant later; it does not repair a bad proposal/observation model. |

Important update: MDA's arXiv v4 metadata explicitly states that prompt leakage in
physics and chemistry was fixed and experiments rerun. This does not establish a
leak in our experiments, but supersedes relying on old family-specific hints. Use
generic context, units, and a public compiler shared by every arm, with no hidden
truth-family identifiers. Inspect v4 rather than adopting the August 15 notes as
current. The proposed changes here are our synthesis, not claims made by MDA.

## Environment choice

| Environment | Role | Main risk |
|---|---|---|
| Compact ChemBench/ActiveSciBench-Chem successor | Primary development candidate: cheap scalar observations, existing compiler, evidence of complementarity | Oracle-bank optimism, weak LLM proposals, oversimplified bins, narrow curated grammar |
| Number Game | Banked positive and inexpensive implementation regression | Saturation, semantic generation failures, limited new novelty |
| BoxingGym/ForceBench | At most one conditional transfer after primary success | No demonstrated multi-depth opportunity yet; avoid another open-ended search |
| NeuronBench, visual Bongard, natural location | Deferred | Cost, likelihood/calibration failures, or weak LLM necessity |

Start with four real measurements, at most eight public assay designs, and d1/d2/d3
contingent policies. These are proposed feasibility settings, not scientifically
validated parameters. Preserve raw scalar observations for inference. Any discrete
branch approximation must integrate the raw likelihood and pass decision-regret
tests; do not throw away the magnitude simply to make the tree finite. Start with
small parameter dimension and quadrature/reference enumeration where possible.
Use source-defined noise and equal assay costs first; heterogeneous costs require
a justified prospective extension, not an arbitrary gate that forces a route.

Truth compositions/parameters, prior construction, prompting examples, and final
test worlds must have explicit separate provenance. Old v4 outcomes are development
data. A new holdout is chosen by rule before new model responses, never by selecting
worlds on which our LLM or planner looks good.

## Architecture to test

1. Public typed expression compiler with dimensional/finite-output checks. The
   LLM proposes complete compatible expressions or minimal patches; it must not
   have to guess a hidden registry name or redundantly encode the same fact twice.
2. Numerical parameter inference and structural weights. Retain useful old models,
   canonicalize duplicates, and freeze a modest model-fit budget. Evidence pruning
   is audited against a larger reference pool.
3. State contains real history, model pool, parameter uncertainty, signed residuals,
   tried edits, remaining budget, and any auxiliary predictor. Names or summaries
   generated from the true mechanism are forbidden.
4. A candidate experiment is evaluated by simulating observations, applying the
   same model-update/proposal interface, and evaluating final predictive loss after
   contingent continuation and the fixed base-policy tail.
5. First validate using direct, memoized proposer responses on a SMALL discrete
   history panel. Memoization must be keyed by the entire public state and proposal
   draw identity. One immutable stochastic realization is not proof of fresh-call
   fidelity: use independent complete banks and fresh held-out proposal draws.
6. If full online branching is unaffordable, learn a transition emulator from
   source-only LLM responses, freeze it, and deploy the SAME emulator in both real
   and simulated updates for the first result. Label this an LLM-derived proposer,
   not live LLM replanning. Live-proposer transfer is a separate experiment with
   multistep action-regret tests. Never silently substitute nearest-neighbor atlas
   predictions for a different live transition process.
7. Incumbent-preserving contingent search uses common random numbers for comparison
   and fresh validation simulations for selection. All model-fit, proposal, search,
   amortization and validation costs are reported.

A small fixed initial LLM-generated ensemble is a diagnostic bridge. If planning
fails even there, dynamic refresh cannot be presumed to fix it. A bridge pass alone
does not establish that anticipating future discovery is the contribution.

## Experiment sequence

### A. Correctness and practical feasibility, before paid execution

Bank the runtime-loss incident; preserve the easy shard and old nulls. Extract only
reusable compiler/inference/planning components. Add independent tests for actual
depth semantics, nested-policy inclusion, complete-policy execution, leakage,
posterior subset interpretation, duplicate invariance, and same-information final
forecasting. Create analytic toy cases with a known myopic trap and a known plateau.
Both must be recognized correctly. Stop after a bounded local working session if
the core planner cannot finish a representative pilot within one hour and a small
durable disk budget. Do not build another full oracle grid first.

### B. Small LLM semantics gate, before expensive opportunity studies

Freeze 24 development histories including noisy, ambiguous, and expanded-pool
states. Use ordinary residuals and executable compositions. Compare a budget
nonreasoning proposer, one stronger nonreasoning proposer, and matched history-blind
and grammar-search controls. Freeze actual endpoint IDs, prices, prompts, caps,
retries, and fallback rules before dispatch. Do not reuse any closed interface.

Suggested development targets: >=95% executable responses after only frozen
syntactic handling; real-history proposals improve held-out prediction relative
to history-blind proposals; and median proposal-induced decision regret below 10%
of recoverable reference improvement. These are provisional design targets, not
newly imposed reinterpretations of old runs. Freeze their precise denominators
and uncertainty treatment before responses. Perfect truth-string recall is not
required if a different expression predicts equally well.

Select the cheapest model that clears the semantic and decision tests. The old
DeepSeek result does not establish the current best model, and the model names in
old automation are not current authorization. Thinking remains a separately
labelled naive baseline unless a new comparison explicitly studies it.

### C. Agent-realistic opportunity and decision-fidelity audit

On 12-24 development worlds, compare oracle opportunity with an agent-built
predictive distribution, including the same-information predictor control. Use the
successful LLM interface now, not only a hypothetical ideal proposer. Measure
depth-specific value bias, selected-action regret, changed roots, and sensitivity
to fresh proposal draws. Estimate opportunity separately from numerical noise and
model bias. A whole-action Spearman threshold alone is insufficient when many
actions are tied; prioritize wrong-selection regret and uncertainty about the best
versus incumbent policy.

For planning feasibility, seek successive mean reductions near 5% with enough
remaining risk after d2. This is a prospective effect-size target, not guaranteed
or an excuse to filter test worlds. If d3 is saturated or model error is larger
than the estimated gain, stop the monotonic claim on this formulation.

### D. Factorial pilot that identifies the contribution

Run three policy depths crossed with two update models: fixed support during
simulated futures and anticipated hypothesis refresh. Both refresh identically
after real measurements where specified; the contrast isolates anticipation.
Add the controls below. Use 24 development worlds and at least two independent
proposal draws as a starting pilot, contingent on a profiled cost calculation.
Do not label repeated noise trajectories as independent world replications.

| Control | Question answered |
|---|---|
| Same-loss greedy, same inference | Is lookahead valuable, independent of EIG-vs-risk objective? |
| EIG greedy | How does the method compare with BED-LLM-style acquisition? |
| Compute-matched wide myopic | Would spending the extra budget on proposals/action coverage/precision suffice? |
| Real-history-only MDA-style updater | Does anticipating discovery add value beyond adapting after observations? |
| Equal-budget larger upfront hypothesis pool | Is apparent future discovery just delayed access to easily available models? |
| History-blind and shuffled-feedback proposer | Is the LLM using evidence? |
| Grammar/beam or symbolic-regression proposer | Is the LLM useful beyond the supplied public language? |
| Same-information auxiliary/full-pool predictor | Is a restricted terminal estimator manufacturing the advantage? |
| Random experimental policy | Is the task informative even without intelligent selection? |
| Naive thinking policy | Behavioral comparison at equal real measurement budget; report compute separately. |

Controls can first run on the pilot. The definitive study retains the necessary
controls that answer distinct causal questions; do not require every old expensive
ablation to finish before evaluating the main uncertainty. Predeclare exactly which
comparisons are confirmatory and which are diagnostic.

### E. One powered, sealed confirmation

Freeze full population/split, model version, prior, action menu, B, branch method,
proposal process, controls, resource caps, and failure handling before outcomes.
Size the study from the world-clustered paired variance of BOTH adjacent-depth
contrasts and the primary compute-matched comparison. A rough normal calculation
n = ((z_(1-alpha*) + z_(1-beta))*SD_difference / target_difference)^2 gives a
starting point; use simulation of the joint family of tests for the final choice.
Do not promise that 64 or 128 worlds is enough. Independent mechanism families,
not just parameter redraws from one law, determine generalization scope.

Primary claim requires lower mean terminal loss at each adjacent depth, simultaneous
confidence intervals excluding zero for both improvements, a prespecified material
total gain, and superiority to the compute-matched myopic control. To claim the
discovery mechanism, also require the anticipated-refresh contrast. Report all
worlds, failures with frozen fallback predictions, ties, raw depth curves, proposal
variance, log score and compute. Use family/world-clustered paired resampling.
No threshold changes, sample extensions driven by observed significance, or test
selection. Partial success supports only the corresponding narrower claim.

### F. Transfer and paper

Only after E, select one source-qualified transfer environment with a new held-out
mechanism family and the same architecture. The main paper needs the fixed-budget
depth curve, productive-compute control, discovery-anticipation ablation, and two
audited examples showing why the deeper first query differs. Include a low-gap
regime to show that the method plateaus when there is no advantage to exploit.
Release a small complete reproduction that finishes without private temporary
directories or an unbounded cache.

## Budget, schedule, and stopping

The last authenticated account read in this investigation was credits 245,
cumulative usage 220.376693994, balance 24.623306006 dollars. No spend has occurred
since the recorded August 15 close. The hard $5 Europe/London account-wide daily
cap remains. No paid call is authorized by this planning document.

Proposed staged allocation from that balance: at most $2 semantics/model selection,
$4 transition/pilot calibration, $3 factorial pilot, $12 confirmation, $3 transfer
or confirmation-sized supporting replication. These are ceilings, not cost forecasts
or a reserve requirement; each stage still needs token/context-based worst-case
exposure and current catalog checks. If power or runtime requirements cannot fit,
report the shortfall before starting confirmation. Unused daily funds do not roll.

Expected work order is one to two focused implementation sessions for A, one small
paid day for B, then a pilot and power decision. Do not promise an end-to-end ETA
before profiling. Stop/reassess if the LLM does not beat a meaningful proposal
control, the agent's predictive model cannot rank consequential decisions, or the
required sample size is unaffordable. At most one source-qualified alternative
environment is considered after a failure; no renewed broad benchmark sweep.

Use durable repository-adjacent run directories, atomically checkpoint each world
and control, bound caches by bytes, and test kill/restart equivalence before long
jobs. Cached tables are disposable acceleration, never the only scientific state.

## Sources

1. [BED-LLM](https://arxiv.org/html/2508.21184).
2. [CA-BED](https://arxiv.org/html/2606.01182).
3. [Action-BED](https://arxiv.org/html/2606.23662).
4. [Deep Adaptive Design](https://proceedings.mlr.press/v139/foster21a.html).
5. [Rollout Algorithms and Approximate Dynamic Programming](https://arxiv.org/html/2212.07998v3).
6. [Why Non-myopic BO Is Promising and How Far Should We Look Ahead?](https://proceedings.mlr.press/v108/yue20b.html).
7. [Adaptive Submodularity, corrected v5](https://arxiv.org/abs/1003.3967v5).
8. [Model Discovery Agent v4](https://arxiv.org/html/2608.09696v4); [revision metadata](https://arxiv.org/abs/2608.09696).
9. [A Probabilistic Framework for LLM-Based Model Discovery](https://arxiv.org/html/2602.18266).
10. [LLM-AutoSciLab](https://arxiv.org/html/2605.24043).
11. [LLM-ACES](https://arxiv.org/abs/2606.25039).
12. [BoxingGym](https://arxiv.org/abs/2501.01540); [official repository](https://github.com/kanishkg/boxing-gym).
13. [Hypothesis generation and updating in large language models](https://arxiv.org/html/2605.05851).
14. [LLMs for Bayesian Optimization in Scientific Domains: Are We There Yet?](https://arxiv.org/abs/2509.21403).
15. [Large Discovery Models v2](https://arxiv.org/html/2608.15669v2).
16. [Large language models as uncertainty-calibrated optimizers for experimental discovery](https://www.nature.com/articles/s42256-026-01283-z).
17. [Robust Experimental Design via Generalised Bayesian Inference](https://arxiv.org/abs/2511.07671).
18. [Bayesian Experimental Design via Score Matching](https://arxiv.org/abs/2607.08335).
