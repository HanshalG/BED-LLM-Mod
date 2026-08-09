# Bongard OpenWorld Closest-Prior Citation Amendment

Date: 2026-08-09 (Europe/London)

Status: **prospective literature-only amendment before any Bongard mechanics,
development, or confirmation response**.

## Reason

The frozen manuscript discussed BED-LLM, CA-BED, ASIG, and other sequential
information-gathering methods, but omitted the closest natural-language
hypothesis-revision precedent:

> Wasu Top Piriyakulkij, Cassidy Langenfeld, Tuan Anh Le, and Kevin Ellis.
> *Doing Experiments and Revising Rules with Natural Language and Probabilistic
> Reasoning*. NeurIPS 2024.

LLM-SMC-S revises low-likelihood natural-language particles after real
observations and selects the next experiment by one-step information gain over
the current particle support. The manuscript now cites this method and states
the narrower distinction: LLM-SMC-S and BED-LLM refresh semantic beliefs between
real observations, while the present target gives a current query prospective
credit for the answer-conditioned generated support available to a later
adaptive decision.

This is a distinction in planning objective, not a priority claim. The paper
does not claim to be the first system that combines LLM-generated hypotheses,
online belief revision, and information-gain experiment selection.

## Exact Scope

The amendment changes only:

- related-work and motivation prose in `paper/main.tex`;
- one bibliographic record in `paper/references.bib`;
- hashes that make the deterministic paper renderer fail closed on either file.

It changes no model, effort, prompt, image, task, seed, candidate action,
likelihood, policy, endpoint, threshold, gate, request count, cost cap,
authorization, result mapping, generated-result wording, or headline rule.

## Rebinding

Historical frozen hashes remain recorded in the original protocol:

- manuscript:
  `67be70e211ad015f3bfe127843c76d9ca7fd61016c07bdbe9ed1ef64a4b1f3c5`;
- references:
  `ac38d0acb02c3328d6020c127a68dc362d6c4d193dbc8e189bed8d0b0c861d4a`.

The prospectively amended files are:

- manuscript:
  `825b74e8030fea6c44406c8ca625345a57dffbded3622e505e9f764601ba88e2`;
- references:
  `44cd1c38638f57e6cbe9c53c3c5d94007a230686ca59949a58f49ed3b0d66768`.

The original paper-fragment protocol remains immutable. The renderer and its
mandatory DINO+SigLIP wrapper must bind this amendment and the amended hashes.
Unknown or mismatched files continue to fail closed.

## Validation

Before rebinding, the amended manuscript passes all 14 required limitation
topics, both required figures, all 38 hash-bound claim bundles, and compiles to
the registered six-page maximum. This amendment made zero model calls, opened
no Bongard label or endpoint, and cost `$0`.
