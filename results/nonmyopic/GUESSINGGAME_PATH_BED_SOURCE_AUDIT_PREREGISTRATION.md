# GuessingGame Path-BED Source Audit Preregistration

Date: 2026-07-29

Status: **frozen before split generation or any new model response**.

## Disclosure

This is a reproducibility and split freeze, not a blind statistical source
confirmation. Exploratory inspection of the public release already established
the approximate counts used to choose the gates below. No development,
confirmation, or new model response has been opened.

## Source

- official repository: `https://github.com/cincynlp/GuessingGame`;
- commit: `df56f1f13fefc4a8ba2c1f89d5026c027f5f42b3`;
- object vocabulary SHA-256:
  `a55a5f9410c8fe3e7cb34e3f57f51b0ad96a6d52db4b261204a48b9685e9fa58`;
- released GPT-4o open-game log SHA-256:
  `a41a4dd3bccc444555c054f362c41e0fe747de79fda9183a893bf195cf0db6d4`.

The repository does not include a license file. This project will not
redistribute its object vocabulary or raw trajectories; committed artifacts
contain only source hashes, aggregate counts, and opaque split identifiers.

## Adaptation

Each eligible released target has two immutable semantic observations:

1. the answer to `What material is the object made of?`; and
2. the answer to a primary-function question.

Rows are excluded if either observation is missing or empty, or if either
answer literally names the target. The prior is uniform over the released
object vocabulary. The two available action sequences expose exactly the same
facts in opposite order:

- material then function; or
- function then material.

An LLM will later receive the released vocabulary under opaque integer IDs and
only its own ordered question-answer history. It will retrieve a bounded
weighted hypothesis belief. The target and unasked target response remain
hidden. Thus terminal differences between the two orders isolate the LLM's
path-dependent semantic belief dynamics; the oracle table and exact target
endpoint are immutable and make no new model calls.

## Frozen Audit Gates

All gates are conjunctive:

- exact source commit and file hashes;
- exactly 858 unique objects and 858 released games;
- game targets equal the object vocabulary exactly;
- at least 800 eligible nonleaking two-action rows;
- both material and function observations retain at least 100 exact-answer
  collisions across eligible targets;
- seed `39400` creates opaque disjoint splits of 5 serving, 10 mechanics,
  32 development, and 64 confirmation targets;
- every selected row remains eligible;
- the public manifest contains no target, question, or answer text; and
- both action orders use the identical pair of released observations.

Failure closes this source route. Passage authorizes only a separately frozen
exact ten-call structured serving smoke for semantic hypothesis retrieval.
It does not authorize mechanics, ranking, policy, or confirmation.

No OpenRouter or OatML call is part of this audit.
