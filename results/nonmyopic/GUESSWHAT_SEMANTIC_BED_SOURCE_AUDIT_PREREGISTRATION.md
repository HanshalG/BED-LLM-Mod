# GuessWhat?! Semantic BED Source Audit Preregistration

Date: 2026-07-29

Status: **frozen before source aggregate or model response**.

## Question

Does the official GuessWhat?! release provide a reproducible, intrinsically
semantic environment for non-myopic Bayesian object identification?

The intended hidden state is one released target object among the annotated
objects in an image. A policy sees the source image with numbered candidate
boxes and asks natural-language yes/no questions. A separate visual likelihood
model estimates answer probabilities over every candidate; an independent
visual oracle supplies the realized answer for the released target. Exact
target-object identification is the external endpoint.

This is a new BED adaptation, not an evaluation of the legacy TensorFlow
GuessWhat?! oracle, questioner, or guesser.

## Bound Source

- official repository:
  `https://github.com/GuessWhatGame/guesswhat`;
- commit:
  `346b7de65d5f18fb8c7d357b7c743d02be429d8a`;
- official test archive:
  `https://florian-strub.com/guesswhat.test.jsonl.gz`;
- archive SHA-256:
  `c26c08fbb860786f25ab6940dab135c4f61a6404d0c89bbf5b7a21716306548c`.

No model may see the released target object ID, human question-answer dialogue,
game outcome, or object category annotations during planning.

## Eligibility

A row is eligible only when:

- the released human game status is `success`;
- it has 5 through 12 annotated candidate objects;
- the target is present and is not a crowd annotation;
- it has at least four structurally valid human question-answer pairs;
- every object has a valid category, ID, and four-number bounding box; and
- at least one category occurs for two or more objects, ensuring that category
  naming alone cannot always identify the target.

Deduplicate eligible rows by source image before splitting. Hash order is
SHA-256 over seed `39000`, image ID, dialogue ID, and full canonical row hash.

Freeze the first 2 unique images for serving, the next 20 for development, the
next 60 for holdout, and leave the rest unused. Public split rows may contain
only dialogue/image IDs, row hash, image filename/URL, and structural counts.

## Frozen Gates

All gates are conjunctive:

- repository commit and archive hash match;
- the archive contains exactly 23,115 rows;
- every released target ID is among its row's object annotations;
- at least 5,000 eligible unique images remain;
- serving/development/holdout sizes are exactly 2/20/60;
- selected dialogue and image IDs are disjoint;
- every selected row remains eligible;
- the candidate payload omits target IDs, human answers, outcomes, and category
  annotations; and
- the public manifest omits endpoint fields and human answers.

Failure closes this source route without eligibility, split, or threshold
repair. Passage authorizes only a separately preregistered exact ten-call
multimodal serving smoke on the two serving images. It does not authorize a
policy, ranking-fidelity test, development endpoint, or holdout access.

Model calls and projected cost: `0` / `$0`.
