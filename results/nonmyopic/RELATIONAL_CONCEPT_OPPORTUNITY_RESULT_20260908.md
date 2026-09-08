# Complete relational opportunity pilot: small positive effects, gate null

Previous turn classification: progress. It supplied the reproducible prior and
independent world sampler required to run this full pilot rather than another
structural inventory. Protocol/runner frozen at69807815 before label computation.
Result SHA256:86b5ad180d2f5a3e4dc6e13907a4827d2eda4cb166e88281000b7fa9f61786f3.

## Results

All four512-draw panels and all controls completed. Four measurements per arm,
eight available queries,64 fixed disjoint prediction targets. Expected terminal
Brier loss is exact under each empirical prior, including repeated draws.

| Panel | h1 / exact myopic | h2 | h3 | Receding open-loop h3 | Random |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | .03787620 | .03657069 | .03687652 | .03932283 | .07278830 |
| 1 | .02992141 | .02867455 | .02867455 | .03026277 | .06578368 |
| 2 | .04101354 | .04037290 | .03950725 | .03999719 | .08114921 |
| 3 | .04901209 | .04714928 | .04712878 | .04842296 | .07242129 |
| Mean | .03945581 | .03819185 | .03804677 | .03950144 | .07303562 |

- Aggregate h1-to-h2 improvement3.2035%; h2-to-h3 improvement0.3799%.
- Aggregate h1-to-h3 improvement3.5712%.
- h3 improves over receding open-loop by3.6826% and random by47.9066%.
- h3 beats h1 on all4 panels; h3 beats h2 on2, ties on1, loses on1.
- Nonincreasing h1/h2/h3 on3/4 panels satisfies that gate, but both adjacent
  aggregate gains miss the separately frozen5% requirements.

Status: `finite_reference_opportunity_null`. Positive descriptive effects are
retained; the null must not be renamed a pass. Four empirical-prior panels are
not a powered population comparison. No thresholds, seeds, grammar probabilities,
world densities or treatment of constant concepts will be changed to rescue it.

## What This Establishes

Unlike the continuous SciLaws work, numerical execution is not the blocker here.
Whole-panel construction plus comparisons took0.48-0.70s,2.16s summed. Cached
policy comparisons took0.027-0.035s per panel, below the5s ceiling. These are
amortized shared-cache timings, not independent per-arm deployment costs.

There is a real observation-contingent advantage over the receding open-loop
control in this finite model. However, added depth contributes little beyond h2.
More importantly, panel0 loses from h2 to h3 with exact rational inference and
no LLM, likelihood approximation, support refresh or Monte Carlo action scores.
Thus eliminating LLM scoring noise would not, by itself, guarantee the user's
desired strictly monotonic deployed-depth curve.

The key distinction is between optimizing a fixed remaining budget and deploying
a truncated receding horizon. h2 and h3 optimize different terminal truncations
when four measurements remain. Their subsequent replanning policies need not be
nested improvements. This is a structural explanation of why monotonicity is not
guaranteed, not a claim that this audit identified every causal root-value change.
Do not conflate deeper internal optimization with an unconditional policy-
improvement theorem. Also do not rename recursive full-budget policy improvement
as ordinary horizon depth: that was already an issue in historical ChemBench.

The finite law had105-128 distinct observed extensions per512 draws;84-106 draws
were constant across all72 sampled worlds. All retained their mass. These are
finite-world statistics, not global semantic counts. Removing those draws after
seeing this result would change the experiment.

## Decision

Do not spend on LLM proposals for this frozen distribution: the complete oracle
opportunity gate failed. Keep the honest smaller positive/adaptivity effects as
supporting numerical evidence, not an LLM-native headline. Close this exact
grammar/world/menu/budget formulation to repeated seeds or parameter rescue.

The unmet dependency is now scientifically clearer: useful semantic complexity
alone does not supply a strong second incremental horizon gain. The next design
must justify a substantial multi-step experimental opportunity independently of
observed outcome selection, while preserving genuinely useful model discovery.
Do not start another language/parser audit or replace this failure with a hidden
truth bank. A retrospective root/continuation analysis may explain this null but
cannot authorize paid work or alter its status. No next experiment is authorized
by this report alone. The full goal remains unachieved.

## Verification And Cost

15 tests passed in0.43s; one optional historical streaming test skipped because
ijson was absent. All six new adapter tests ran, including duplicate mass, fixed
target columns, constant-particle retention and independent general-planner
agreement at h1/h2/h3 for three synthetic seeds. Scoped lint passed.

The outcome file was reserved before computation and all four rows checkpointed.
No raw formulas or world-level labels are in the public result. Upstream pinned
semantics generated new synthetic labels, not released benchmark outcomes.
Model calls0, cost$0. Authenticated account245/220.376693994/24.623306006;
London Sept8 spend0, remaining$5. Process exited successfully, no cluster use,
automation remains paused.
