# RegretBench Release Source Audit Preregistration

Date: 2026-07-28

## Why This Is A New Source State

The prior source audit found the repository cited by RegretBench unavailable. The
paper now links a populated official repository:

- repository: `https://github.com/ngocminhta/RegretBench`;
- observed `main`/`HEAD`: `5e105c6bf033a7261ab5a7fd6f581dbf2d76790a`;
- code license: MIT;
- released datasets: `OpenDomainQA` and `ProductRecommendation`, each with its own
  data-card licensing boundary.

The paper and repository describe a clarification interaction graph (CIG) with
hidden intents, semantic clarification actions, structured observations, Bayesian
belief updates, a model-backed semantic responder, and a finite reference planner.
This is a materially different and newly executable source state, so it may be
audited once without reopening the prior 404 result.

## Scientific Admission Question

The source is admissible as a new LLM-native non-myopic BED route only if all of the
following are true:

1. the official release validates and reproduces its published checksums;
2. a source-clean cohort can be frozen before task values are inspected;
3. the released runtime exposes an executable hidden-intent transition and external
   terminal-intent endpoint;
4. depth-two planning has measurable value over a matched one-step policy under a
   frozen observation model and cost profile;
5. the LLM supplies a load-bearing semantic object that cannot be replaced by merely
   loading a complete finite intent-by-facet response table.

Condition 5 is deliberately strict. The existence of natural-language labels alone
does not make a finite CIG irreducibly LLM-native. Qualifying mechanisms include
history- or wording-conditioned semantic likelihoods, path-dependent support
generation, or a natural-language action space whose useful alternatives are not
already exhaustively enumerated by the release.

## Access Boundary

Before opening task prompts, intent descriptions, facet values, reference questions,
or terminal answers:

1. clone only official commit `5e105c6bf033a7261ab5a7fd6f581dbf2d76790a`;
2. verify repository and dataset checksums;
3. inspect schemas, runtime code, defaults, and split metadata only;
4. freeze a deterministic split from `OpenDomainQA/train`, never from either
   release's non-pristine `test` split;
5. record ordered IDs, source strata, counts, and hashes in a public manifest without
   emitting task text or hidden values.

Seed `24419` is fixed for any source-stratified partition. The intended partition is
`50` mechanics, `500` opportunity, `200` development, `1,000` confirmation, with all
remaining rows retained, subject only to there being enough schema-valid train rows.
If source strata make exact counts impossible, fail and amend prospectively before
content access; do not silently resample.

## Zero-Call Structural Gate

After the manifest is committed, only mechanics and opportunity values may be opened.
The audit must report:

- intent and supported-facet count distributions;
- whether all semantic actions are available at every turn;
- whether responder likelihoods depend on dialogue history beyond current belief,
  selected facet, hidden intent, persona, and surface question;
- whether the reference planner uses cached finite tables or live semantic calls;
- one-step and depth-two reference values under the exact same cost/persona model;
- root-action changes and strict depth-two gains over one-step;
- whether any gain comes from a genuine contingent second action rather than simply
  permitting one more independent clarification.

The opportunity cohort passes only if:

- at least `100` rows have at least `3` intents and at least `2` supported facets;
- at least `30` rows change their root action between matched one-step and depth-two
  planning;
- at least `20` rows have strict depth-two terminal-value gain of at least `.02`;
- mean depth-two gain over eligible rows is at least `.01`;
- at least `10` rows require different optimal second actions after different first
  observations;
- at least one load-bearing LLM-native condition from admission condition 5 holds.

All thresholds are conjunctive. Failure closes RegretBench as a headline route before
OpenRouter use. A structurally positive but finite-table result may be recorded only
as supporting evidence.

## Conditional Paid Work

No model call is authorized by this document. A full structural pass permits a
separate, response-blind preregistration for:

1. a ten-call serving test of the semantic parser/responder and strict codec;
2. a small development comparison of a non-myopic LLM policy, compute-matched myopic
   policy, fixed semantic reference planner, and random control;
3. a powered paired confirmation only after serving and development gates pass.

There is no protected cost reserve. Spend is still staged by scientific gates so that
the available OpenRouter balance is used on informative evidence rather than invalid
mechanics. OpenRouter is the only execution backend; OatML, Slurm, and SSH are
prohibited for this route.

