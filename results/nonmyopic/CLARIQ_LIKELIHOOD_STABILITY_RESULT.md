# ClariQ GPT-5.4 Likelihood-Stability Result

## Decision

The frozen target-free gate failed. The exact five-way `Y`/`N`/`U` likelihood
interface is closed, and no target, answer transition, retrieval endpoint, or
efficacy split was accessed.

## Serving

| Metric | Result |
|---|---:|
| Physical requests / HTTP attempts | `10 / 10` |
| Retries / reasoning tokens / forced exits | `0 / 0 / 0` |
| Parsed maps | `10 / 10` |
| Cost | `$0.006340` |
| Unique base maps | `4 / 4` |
| Informative base maps | `4 / 4` |
| Base entropy range | `.554518` nats |

The four base maps were:

| Question | Base map | Entropy |
|---|---|---:|
| `Q00796` | `UNYUN` | `1.054920` |
| `Q01384` | `NUUNU` | `.673012` |
| `Q03514` | `NNNYN` | `.500402` |
| `Q03741` | `YNUUN` | `1.054920` |

## Failed Gate

Every exact repeat had to agree. Two questions did not:

- `Q01384`: `NUUNU` then `YNNNN`;
- `Q03514`: `NNNYN` then `NUUNU`.

Questions `Q00796` and `Q03741` were stable, including four identical independent
requests for `Q00796`.

## Interpretation

ClariQ cleanly separates two prerequisites that were confounded in autonomous
question generation. Its human-authored question bank supplies diverse semantic
actions and its official graph contains a verified depth-two retrieval gap. Those
properties passed here: all four roots induced distinct informative partitions.
The remaining failure is specifically the LLM likelihood model. Even GPT-5.4 at
temperature zero did not define a repeatable response partition for ambiguous
facet-question pairs.

The disagreement is not harmless sampling variance in a Monte Carlo estimator:
it changes the simulated observation model itself. Deeper EIG would optimize a
different tree depending on which nominally identical likelihood call happened
to be returned. Consensus, majority voting, or prompt repair would define a new
method and cannot be introduced after observing this gate.

Public result:
`results/nonmyopic/clariq_likelihood_stability/clariq-likelihood-stability-20260725T143540Z/STABILITY.json`.
Raw responses remain private and hash-linked from that result.
