# Human Semantic Norms Object Game: Zero-Call Opportunity Preregistration

Date frozen: 2026-07-29

## Question

Can a natural semantic concept-learning environment support a genuine
non-myopic first link before any LLM serving is attempted?

The environment uses human behavioral production norms as an independent
endpoint. The eventual LLM policy, if authorized, must generate and regenerate
open-ended English concept hypotheses over the selected objects. It will not
receive the human feature labels or feature bank. Thus the LLM remains
load-bearing for the usable belief support, while the endpoint is neither
invented by this project nor judged by another LLM.

This first stage is a zero-call source and opportunity gate. It cannot establish
an LLM-native policy result.

## Source

- Repository: `https://github.com/AaltoImagingLanguage/Norms`
- Commit: `d2b940472bb9e88983a5c087594022939cac8895`
- Publication: Kivisaari et al., *Semantic feature norms: a cross-method and
  cross-language comparison*, DOI `10.3758/s13428-023-02311-1`
- `correspondence.csv` SHA-256:
  `e922d57ff87fc6c8a4f2276a589c1645386c0e8f296eca6722998dae35763821`
- `vocab.csv` SHA-256:
  `a9b4ef36a118e1a23371597e673f20cbb058a00472e8c4cbf5703e3b2154156d`
- `vectors.csv` SHA-256:
  `41b620c391b9147a12d63dc9ed1af619e636bd6db4532e56c71a1b144bc54f99`
- `features.csv` SHA-256:
  `1d838554bed2dce20adac939d543d87c81947288801367ed568cfd7fc61adf88`

The paper is CC BY 4.0, but the data repository has no explicit license file.
The raw matrix and Finnish feature strings therefore remain in an external
checkout and will not be redistributed by this repository.

## Frozen Construction

1. Match each production-vector row to exactly one correspondence row through
   `aaltoprod`.
2. Keep unique English nouns with a nonempty, non-abstract category and no
   Finnish or English homonym flag.
3. Rank categories by eligible count descending, then name ascending. Keep the
   eight largest categories having at least four eligible objects.
4. Within each category, rank objects by SHA-256 of
   `40400|category|english_name|finnish_name` and keep four.
5. Preserve category-rank and within-category hash order, yielding exactly 32
   queryable English objects.
6. Turn each human feature column into a 32-bit membership extension using
   strictly positive production frequency.
7. Keep extensions containing 3 through 29 objects and deduplicate identical
   extensions. The prior is uniform over the resulting distinct extensions.

Feature labels are never used for selection, planning, or reporting.

## Exact Policies

A query asks whether one selected object belongs to the hidden feature. Answers
are deterministic memberships. At every reached history:

- greedy EIG chooses the query minimizing one-step expected posterior entropy;
- exact depth `d` chooses the adaptive query tree minimizing expected terminal
  posterior entropy after `d` questions;
- ties use fixed selected-object order.

Both policies adapt after every answer. Terminal multiclass Brier Bayes risk is
reported for the entropy-selected trees as an independent proper-score
endpoint.

## Frozen Pass Gate

Every check must pass:

- at least 96 eligible source objects;
- exactly 32 selected objects from exactly eight categories;
- at least 64 distinct valid human feature extensions;
- at least 16 extensions containing 8 through 24 selected objects;
- the exact depth-three root differs from the adaptive greedy-EIG root;
- depth-three expected terminal entropy improves by at least `0.005` nats;
- depth-three expected terminal Brier risk improves by at least `0.001`.

Depth-one and depth-two results are descriptive. No threshold, source split,
selection seed, support range, prior, objective, or tie rule may be changed
after reading the result.

## Decision Rule

A full pass authorizes a separate, response-blind exact 10-call serving
preregistration. That smoke will test whether a chosen nonthinking planner can
produce valid semantic bitsets on this externally selected universe and whether
history-conditioned regeneration is dynamic and truth-covering under human
targets.

Any failure closes this exact source construction before OpenRouter calls. No
alternate seed, category subset, feature weighting, or support threshold is
allowed as a repair.

## Budget

This stage uses zero model calls and costs `$0`. OpenRouter displayed
`$12.757678083` immediately before freezing this protocol; the user's latest
top-up was not yet reflected. The user has authorized aggressive spending over
four days with no reserve, but only a full gate pass can authorize paid serving.
