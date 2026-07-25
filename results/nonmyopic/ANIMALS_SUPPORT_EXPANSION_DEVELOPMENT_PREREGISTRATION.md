# Animals Support-Expansion Development Preregistration

Date: 2026-07-25

## Motivation And Disclosure

This is a distinct development experiment on the already-open 20-target
stratified Animals development split. It is not confirmatory evidence.

A zero-call post-hoc screen compared four deterministic functions of the
cached regenerated branch supports: expected support size, expected log
support size, negative expected log support size, and Yes/No branch-union
size. Expected size and expected log size made identical selections. Expected
support size achieved mean target-measurement coverage `.515300` versus
`.419050` for immediate EIG, a gain of `+.096250` with `9/8/3`
wins/ties/losses. This result selected the method and cannot be used as an
independent endpoint.

The hypothesis is that a question can be valuable because it causes the LLM
to regenerate a broader next-turn semantic support. Immediate EIG over the
current generated names cannot value that transition.

## Aligned Environment

- Source prior: the frozen seed-24285 126-animal eight-stratum target pool,
  SHA-256
  `f1357649054e41150580201b8ad2318e30fb752ae7adb647e85873f7115a6715`.
- Development targets: the already-open frozen 20-target development block.
- Model: `google/gemma-4-26b-a4b-it`, non-thinking, temperature zero for
  semantic tables and `0.7` for diverse generation.
- One fixed target-independent prehistory question is selected by state index.
- Beliefs are regenerated and filtered through the same eight taxonomic
  strata used to sample the target prior.
- Three retained root questions are generated from five requested candidates.
- Both Yes and No branches are regenerated for every root.

For every prehistory or root question, one stateless environment call labels
all 126 prior animals Yes/No before the hidden target row is read. The same
table supplies semantic likelihoods and the realized answer. Candidate and
belief-generation prompts receive no hidden-target field. The environment and
policy use the same model family but separate stateless calls.

## Paired Arms

Every arm shares the generated current belief, three questions, six
counterfactual regenerated supports, semantic tables, and hidden target.

- **Support expansion:** maximize
  `p(Yes) * |support_Yes| + p(No) * |support_No|`.
- **Immediate EIG:** maximize one-step EIG on the current generated support.
- **Support retention:** maximize expected retained probability mass from the
  current support.
- **Branch union:** maximize the deduplicated union of the two branch supports.
- **Random:** seeded uniform root from the same three.

The primary development endpoint is realized hidden-target inclusion in the
support corresponding to the aligned target answer. Model-averaged expected
truth coverage is secondary. Uniform target mass conditional on inclusion,
realized support size, candidate ranking, and recovery after initial omission
are diagnostics.

## Stages And Gates

The exact two-target serving smoke is mechanics-only. It must complete both
states with three candidates, binary aligned answers, dynamic support scores,
zero reasoning/forced exits, and cost at most `$0.50`.

Only a passing smoke authorizes the exact 20-target development run. Development
must satisfy all of:

1. all 20 states complete with three candidates;
2. support scores vary on at least 10 states;
3. support expansion and EIG select different roots on at least five states;
4. EIG realized coverage is between `.05` and `.95`;
5. support expansion has positive realized and model-averaged coverage gain;
6. realized wins exceed losses;
7. support-score pairwise accuracy on realized coverage exceeds EIG;
8. support expansion recovers at least one initially omitted target;
9. zero reasoning/forced exits and total development cost at most `$2.00`.

Failure closes this exact interface. Passing authorizes a separate
preregistration, before opening any of the 60 holdout endpoints, with a
strictly positive paired uncertainty criterion and no threshold tuning on the
holdout.

No OatML resources are used. OpenRouter calls may start only while both the
authenticated balance and project ledger preserve the protected `$25`
reserve.
