# Third pass: a smaller architecture and an identifiable sequential claim

Date: 2026-09-08 (Australia/Melbourne).
Planning-only follow-up to NONMYOPIC_RESEARCH_SECOND_PASS_20260908.md.
No new model calls, environment outcomes, experiment authorization, or changes
to frozen study statuses. Inspected code at commit
`26c10df34c9f9e19f46c8cc7ef34488efceca08e`.

## Decision

Build towards genuine receding-horizon BED over LLM-generated executable models,
but first separate three problems we have been trying to solve simultaneously:

1. Can the LLM build an evidence-responsive, predictively useful model space?
2. Does observation-contingent h=1/2/3 planning improve decisions in that space?
3. Can anticipating future LLM computation add value beyond 1 and 2?

The immediate architecture should refresh models after REAL observations and hold
their structures fixed within each numerical planning pass. Posterior weights and
parameter uncertainty still update along imagined branches. Optimize every
contingent decision inside the declared horizon and replan after every real
measurement. Introduce prospective simulated structural refresh only after that
instrument works. This is a narrower intermediate result, not completion of the
strongest path-dependent-discovery goal in GOAL.md.

ChemBench remains the first candidate, conditional on cheap source-grounded
tests. Stop adding architectural layers or environments before resolving these
questions. A polished plan cannot substitute for an end-to-end pilot.

## 1. New code finding: Number Game optimized roots, not full future policies

In `scripts/number_game_depth_three_development.py:101`,
`evaluate_policy_root_depth_three` takes a prescribed root. It chooses query two
with `best_query(first_support)` at line 117 and query three with
`best_query(second_support)` at line 127. `best_query` optimizes immediate EIG.
There is no optimization over alternative two-step continuations at query two.

In `scripts/number_game_crossfit_depth_three_confirmation.py:214`, the scorer
selects a root using either two-query or three-query predictive risk. At line 222
it evaluates all roots with the SAME three-query continuation function, and at
line 245 maps the named policies to those per-root endpoints. Thus measurement
budget is matched; this is not a three-versus-two-real-measurement comparison.
But the changed treatment is ROOT SELECTION using a longer greedy rollout, not
an h-step optimizer deployed at every real turn. Qwen imports this scorer, and
the canonical pooled report combines those source comparisons.

This is legitimate non-myopic rollout evidence. It is not evidence that all
future actions were optimized at h=3, nor that h=3 was reapplied after each
observation. The old ChemBench ladder has a different depth issue: recursive
full-budget policy improvement. Neither should be relabelled as the new h.

The saved Number Game branch generation at lines 448 onward also calls the LLM
only for the second query chosen by greedy EIG in each first-response branch.
Therefore the saved dynamic tree does not, by itself, identify a full optimal
contingent-policy replay. Missing alternative-history proposals cannot be
invented, treated as empty supports, or supplied by inspecting hidden targets.
A static-union replay would be a new retrospective diagnostic with a different
updater, not a reconstruction of an unrun live policy.

Additional endpoint caveat: Number Game Brier excludes the queries selected by
each policy (line 151 onward), which makes its target set policy-dependent.
That is a defined "predict the remaining numbers" task, not automatically an
invalid score. For the NEW prediction study, use one independently drawn query
distribution with fixed weights and no policy-specific deletion of difficult
targets. Preserve historical results and describe their actual estimand.

Source hashes:
- `number_game_depth_three_development.py`: `593f039aa07ded3e930fd834a130ba28d85a885c1e8da50257ae96728857d0c1`.
- `number_game_crossfit_depth_three_confirmation.py`: `9b4237d303748c22a99f515f259424a64a1a61d18e249fa02080e6ee035a3fdf`.
- `number_game_generator_aware_bed.py`: `1df1eab2d6e7dbbebb89e31d6c9863160d15ce519c8d9c55724de577a500a35a`.

## 2. The key distinction: information versus computation

Let H be the real history, K the fixed background knowledge, theta the physical
world, and U a proposer output generated from H,K and independent randomness.
If the proposer gets no additional world observations, then under this stipulated
generative model:

    theta is conditionally independent of U given H,K.

Consequently U cannot improve an already exact Bayes predictor conditional on
H,K. This does NOT say LLMs are useless or cannot supply useful prior knowledge.
K includes that knowledge in this thought experiment; extracting and applying
it is precisely what a computationally limited agent cannot do exactly.

For squared loss, this gives an instructive decomposition for a scalar target F:

    E[(g(H,U)-F)^2 | H,K]
      = Var(F | H,K)
        + E[(g(H,U)-E[F | H,K])^2 | H,K].

The first term is uncertainty that measurements can reduce. The second is the
excess risk of the bounded predictor, which better model discovery/computation
can reduce. A measurement can affect BOTH: it can reveal the mechanism and make
the right explanation easier for the LLM to generate with a limited budget.
This identity is an explanatory ideal-model calculation, not an estimated
decomposition of our current M-open runs or a novel theorem.

The strongest project claim is thus resource-bounded sequential BED: choose
measurements that improve future prediction, accounting for how data changes the
usefulness of future inference computation. It requires separate physical and
computational budgets. It cannot be certified by entropy reductions in a changing
finite hypothesis pool. Classical rational metareasoning already supplies the
value-of-computation framing; our contribution would be a tested implementation
for LLM-generated scientific models, not inventing that framing.
[Russell and Wefald](https://doi.org/10.1016/0004-3702(91)90015-C).

Refinement to pass two: equal information access does not require every practical
predictor to be computationally omniscient. It requires no private truth bank or
arbitrary ban on already computed mechanisms, and fair charged fitting/inference
limits. A full-information Bayes predictor is a diagnostic reference, not a
baseline allowed unlimited hidden computation while other arms are constrained.

## 3. One more necessary distinction: lookahead versus adaptivity

An h-step plan can improve over greedy simply by selecting a better SET of
experiments. That does not establish that conditioning the second and third
actions on their predecessors' outcomes is important.

Under a common model, horizon, and feasible action set, compare:

    J_open(h) = best expected risk of an h-action precommitted sequence
    J_tree(h) = best expected risk of an h-step contingent policy
    adaptation_gap(h) = J_open(h) - J_tree(h) >= 0.

The inequality follows because a contingent policy can copy any fixed sequence.
It need not be strict and does not guarantee empirical monotonicity across
receding-horizon h at fixed real budget. It assumes the same forecast/update
machinery, with no action-dependent computational advantage smuggled into an arm.

Add an OPEN-LOOP-LOOKAHEAD h3 control: future queries cannot depend on imagined
answers within its planning pass, but it still replans after every REAL observation.
This isolates anticipated adaptivity from actual feedback use. It is not the
same as committing to all B queries at the beginning of the episode; label the
latter separately if included. One-shot multi-step BO explicitly distinguishes
adaptive trees, fixed-base-policy rollouts, and nonadaptive approximations, and
reports strong performance from cheaper approximations too.
[Jiang et al.](https://arxiv.org/html/2006.15779).

Important environment diagnostic: with a known linear observation model,
Gaussian prior, known Gaussian noise, action-only constraints, and squared loss
for a fixed linear target, posterior covariance depends on designs but not
observed values. Optimal fixed-budget design then needs no answer-conditioned
adaptation, even when jointly choosing experiments can beat greedy selection.
This conclusion follows from covariance-based Bayes risk: every adaptive path
has the risk of some feasible fixed design, so its expected risk cannot beat the
best such design. Unknown noise/hyperparameters, nonlinear targets, different
losses, or state-dependent feasibility can break the argument.
See the linear-Gaussian Bayes-risk foundations in
[Alexanderian et al.](https://arxiv.org/abs/1408.6323).

Therefore a highly Gaussianized chemistry approximation may make the desired
adaptivity gap vanish. We should test mechanism ambiguity and outcome-dependent
continuations, not merely increase noise or add more parameters. Preserve
multimodal predictive structure where it matters. Changed action IDs alone are
insufficient; alternative branch actions must have materially different risk.

## 4. Recommended first architecture

Use an MDA-style proposer plus numerical inference, but keep the first planner
smaller than the previous designs. MDA already supports executable proposal,
evidence, model averaging, and adaptive support expansion; these components are
borrowed foundations, not our novelty.
[MDA v4](https://arxiv.org/html/2608.09696v4).

1. After each real observation, run the same bounded evidence-conditioned proposal
   and parameter-fitting procedure for every method. Keep the full valid-model
   registry; active-pool pruning is a common numerical resource rule. Fix model
   priors independently of the current outcomes and label finite-pool Bayes as
   conditional on that pool.
2. Compile the current weighted model/parameter particles into a shared numerical
   observation and target predictor. Parameter uncertainty is essential; do not
   turn each mechanism into a point estimate simply to make branching cheaper.
3. Within a planning pass, freeze STRUCTURES and update particle weights using
   full likelihoods after each simulated raw observation. Search all designs on
   the small menu. Optimize future actions using the same predictive loss, not
   EIG as a hidden continuation policy. Simulated branch outcomes never enter
   the real history or serve as new evidence for the physical world.
4. For h1/h2/h3, execute the selected first action, collect the actual observation,
   refresh from real history, and replan. Only min(h, remaining B) decisions are
   optimized. Same B, same target distribution, same tie-breaking rule.
5. Compare root values with independent numerical refinements and paired samples.
   Count numerical conditioning operations and fits as well as tokens and calls.
   No LLM call is made per numerical branch in this first architecture. Calls
   scale with real-update schedule, not exponentially with h; total token use
   can still differ because histories and triggers differ and must be reported.

This deliberately does not forecast future STRUCTURAL refresh. It can therefore
misestimate what its real future updater will do. Measure that first-link error;
do not call its simulation an exact model of deployment. Starting here isolates
whether actual-horizon planning is viable before adding the still unverified
stochastic discovery-transition model. A pass demonstrates lookahead with adaptive
LLM-built beliefs, not anticipation of previously unrepresented discoveries.

For a later anticipatory extension, retain pass two's immediate-reuse rule. But
if giving greedy the speculative union erases a gain, do not infer that lookahead
is intrinsically useless: the pool changed too. Recompare h1/h2/h3 on the SAME
union at common decision states and measure the interaction. At each arm's own
visited real history, any shared proposal generator must be runnable there;
never transplant another arm's actual future observations or their proposals.

## 5. Risk evaluation must not reward self-reported certainty

For a fixed planning pass, sample physical worlds from an agent-accessible joint
predictive model, keep each latent world persistent along its rollout, and score
terminal predictions against its simulated target values. An evaluator can know
that sampled world; the action selector and proposer cannot. All candidate
policies share the same outer target definition and predictive world law.

Only when branch updates are coherent exact conditioning and the final predictor
is the corresponding Bayes action may expected terminal squared error be
replaced by expected posterior variance without changing the objective. An
approximate support refresh can become more confident while being less accurate.
Scoring its own entropy/variance then confounds confidence with utility. The
Number Game evaluator already scored against separate target hypotheses; preserve
that separation rather than regressing to changing-support entropy differences.

None of this turns a misspecified simulator into ground truth. Use source-only
held-out predictive calibration, approximate-versus-refined action regret, and
prospective deployed endpoints as separate checks. Do not certify epistemic
accuracy by agreement with the same misspecified model at higher compute.

## 6. A bounded work package, not another sprawling grid

The next implementation package should have exactly three deliverables:

**A. One complete numerical reference.** A tiny arbitrary-horizon solver and
independent tests for a myopic trap, a zero-adaptivity linear-Gaussian case, a
positive-adaptivity discrete case, and a legitimate depth plateau. Test that
sibling histories share decisions until observations distinguish them. Store
actual planned trees and their maximum optimized depth. No empirical efficacy
claim comes from these constructed tests.

**B. One source-grounded chemistry development panel.** Start with eight worlds
as an engineering pilot, not a powered result. Cover prespecified mechanistic
ambiguities and a low-opportunity regime under a source-defined sampling rule.
Keep raw scalar observations, a small full assay menu, and equal measurement
costs initially. Freeze exact split, priors, noise, B, losses, and runtime ceilings
before new responses. Four measurements and at most eight designs remain
feasibility proposals, not settings to tune until the curve is monotonic.

The numerical pass compares true-population reference opportunity separately
from deployable-prior opportunity. It must complete a whole world plus controls
and checkpoint within a predeclared local runtime/storage cap. No unbounded
SQLite transition archive and no medium/hard sharding before a full pilot works.
Reusable mathematics need not be rebuilt as a full discovery platform.

**C. One small new LLM gate and paired pilot.** Require useful held-out prediction
from executable proposals and real-history improvement over history-blind and
symbolic-search proposals. Then compare h1/h2/h3, open-loop-lookahead h3,
productive-compute-matched h1, and random. Naive thinking is a separately labelled
behavioral baseline. This is a maximum of six core numerical policy arms before
the naive comparison, not a ten-way API-heavy discovery grid. No automatic
reopening of a closed model interface or old environment endpoint is authorized.

Report raw adjacent-depth paired differences; open-loop versus tree difference;
real-history proposal benefit; model calibration; proposal/inference failures;
and measured cost per complete paired world. Include an example in which an
observed outcome changes the useful NEXT assay, with numerical value differences,
not just a plausible LLM narrative. Publishability depends on those mechanisms
and controls, not only on obtaining three decreasing point estimates.

The subsequent decision is finite:
- No deployable numerical opportunity: stop this formulation; do not buy proposer
  calls or alter noise/targets after seeing outcomes to manufacture headroom.
- Opportunity but inadequate proposals: focus only on the proposer interface;
  more horizon is not the remedy.
- Good proposals but unstable action values: refine numerical inference/planning
  against decision regret, without changing the scientific task.
- Better h3 than h1 but h2-to-h3 unresolved: size from that weaker paired effect
  or narrow the claim; neither a new seed nor a more flattering metric repairs it.
- Tree ties open-loop lookahead: retain any valid non-myopic result, but do not
  claim that anticipated branching caused it.
- All main links pass: freeze one powered fresh-family confirmation; only then
  test the stronger anticipated-discovery extension and a single transfer.

The $5 Europe/London account-wide limit and no-cluster instruction remain. No
fresh account balance was asserted or paid request attempted during this pass.
Sample size and spending require measured full-arm costs and current authenticated
account/catalog checks, not the earlier speculative $24 allocation. Automation
remains paused. No frozen gates or scientific result files are edited.

## Final assessment

The prospect remains interesting, but our next paper claim should be earned in
layers. A clean non-myopic result over genuinely useful LLM-generated model spaces
is more credible than attempting to solve open-ended Bayesian inference,
counterfactual LLM behaviour, adaptive tree search, and strict depth monotonicity
at once. It still requires an LLM proposal advantage under fair resource limits.

The strongest goal remains experiments that improve the effectiveness of future
LLM inference, with a demonstrated additional value from anticipating that effect.
The first architecture is an instrument for reaching that goal, not a declaration
that a simpler hybrid has already achieved it. Strict monotonicity is an empirical
target under a stated population and budget, never a universal promise.
