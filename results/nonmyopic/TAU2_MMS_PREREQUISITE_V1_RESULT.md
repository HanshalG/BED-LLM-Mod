# Tau2 MMS Prerequisite Ranking V1 Result

Date: 2026-07-24

Status: closed; no policy experiment authorized.

## Serving Smoke

The Gemma 4 26B A4B non-thinking interface completed exactly 18 physical
requests with zero reasoning tokens, retries, forced exits, or runtime errors.
Cost was $0.00477317.

The initial coverage scorer incorrectly required an `unknown` structured field
to equal an official `normal` field. A preregistered pre-formal amendment
corrected this as a zero-call, hash-locked recovery: official faulty fields
still require `faulty`, while official normal fields accept `normal` or
`unknown` but reject `faulty`. The recovered smoke passed with mean official
signature coverage 5.5/6. Every setup rollout chose the legally unlocked
messaging-permissions diagnostic.

Original smoke artifact SHA-256:
`389aaf02038113a4dd486b58870f26b14ae046d3bd5119bbfba41a587e8b4ff4`.

## Formal Failure

The formal stage issued exactly 108 physical requests with zero reasoning
tokens and cost $0.03115662. It failed closed before endpoint scoring:

- Case 3, APN root: the rollout invented an illegal `storage_permission`
  action before app discovery.
- Case 11, Wi-Fi-calling root: hypothesis `h8` was assigned to both the
  `enabled` and `unknown` branches.

These are substantive policy-tree inconsistencies, not harmless formatting
aliases, so neither response was repaired or resampled.

## Zero-Call Diagnostic

For diagnosis only, both invalid action rows were marked invalid and the
remaining 94/96 action rows were evaluated unchanged:

| Metric | Value |
|---|---:|
| Mean official semantic support coverage | 5.5 / 6 |
| Valid-score Spearman vs exact depth-2 value | -0.0139 |
| Depth-2 setup selections | 0 / 12 |
| Depth-1 setup selections | 0 / 12 |
| Depth 2 beats depth 1 | 2 / 12 |
| Mean depth-1 top-1 regret | 0.5486 nats |
| Mean depth-2 top-1 regret | 0.5139 nats |
| Mean regret improvement | 0.0348 nats |

Even ignoring serving invalidity, the semantic rollout scorer did not rank the
known prerequisite opportunity. The model generated broad fault coverage but
overpredicted distinctions from direct diagnostics and underweighted the
multi-outcome permission read. This repeats the project's central finding:
semantic support coverage is not sufficient; compositional likelihood ranking
must align with the environment.

## Decision

Close this exact MMS app-permissions prompt and scorer. Do not tune or rerun it.
The next Tau2 route uses a distinct, larger prerequisite: customer lookup
reveals the line identifier, after which line details distinguish hidden
account states. Its zero-call exact gap is 1.1101 nats rather than 0.3749.
