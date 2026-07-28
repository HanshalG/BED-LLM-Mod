# DiscoverPhysics Fixed-Initial Branch Replication Result

## Outcome

**Replication null on the LLM-support claim; clean replication of the
non-myopic root advantage.**

The sole preregistered run made exactly eight fresh GPT-5.4 branch calls from
the public structured-V3 initial support and branch partition. Phase A passed
all transport, support, breadth, diversity, and internal-selection gates, so
the new 384-map endpoint was opened.

On that endpoint, modular non-myopic B beat myopic D and random A, but the fresh
LLM branch support was indistinguishable from same-root fixed B:

| Policy | Fresh-map trajectory MSE |
|---|---:|
| Random A, modular | 4.788695 |
| Myopic D, modular | 3.481325 |
| Non-myopic B, modular fresh support | 3.049085 |
| Same-root B, fixed support | 3.048537 |

The B-versus-D reduction was `12.416%`, with paired
`MSE(D)-MSE(B)` interval `[0.25913, 0.60942]`. B beat random by
`36.327%`. Nearest-support risk improved by `15.645%`.

Against fixed B, however, the relative reduction was `-0.01798%`, and the
paired `MSE(fixed B)-MSE(modular B)` interval was
`[-0.00975, 0.00880]`. Both preregistered same-root support gates failed.

## Phase A

- Exact requests / HTTP attempts: `8 / 8`
- Retries, reasoning tokens, forced exits/finals: `0 / 0 / 0`
- Cost: `$0.114525`
- Fresh supports changed from initial: `8 / 8`
- Roots with branch-distinct supports: `4 / 4`
- Every refresh: exactly two maps per region and at least three geometries
- B branch continuations: `r4.5_a5`, `r4.5_a2`
- Immediate EIG: D `1.405768`, B `1.276714`
- Modular internal risk: B `0.285809`, D `0.592081`
- Predicted B-versus-D reduction: `51.728%`

All Phase-A gates passed before endpoint access.

## Regional Diagnostic

This zero-call decomposition uses the already-open endpoint and does not alter
the result:

| Region | Fixed B MSE | Modular B MSE | Fixed minus modular | Modular wins |
|---|---:|---:|---:|---:|
| NE | 2.399628 | 2.452869 | -0.053242 | 42 / 96 |
| NW | 2.156253 | 2.169470 | -0.013217 | 41 / 96 |
| SW | 5.754303 | 5.680682 | +0.073621 | 91 / 96 |
| SE | 2.909490 | 2.809597 | +0.099893 | 96 / 96 |

The branch support helps the lower-prior SW/SE regions but harms the
higher-prior NE/NW regions. Under the frozen `.4/.3/.2/.1` prior, these effects
cancel to a weighted fixed-minus-modular difference of `-0.000548`.

This is stronger evidence than another full-tree failure for where the problem
lies: breadth-preserving branch generation improves coverage, but a fixed
global component weight does not calibrate its predictive value across semantic
regions.

## Interpretation

Holding the initial LLM belief fixed removes initial-tree instability and
reproduces the non-myopic root decision on an independent endpoint. It does not
rescue the claim that branch-conditioned LLM support adds outcome value beyond
choosing that root.

Across the available evidence:

- the first tree showed `+2.41%` support value;
- a fresh structured tree showed `-5.66%`;
- this fixed-initial, fresh-branch isolation shows `-0.018%`.

The robust result is the non-myopic B-versus-D root advantage under exact
likelihoods. The LLM-native support increment remains generation- and
calibration-sensitive.

No rerun, alternate component mass, regional weight, endpoint subset, or second
branch sample was inspected.

## Provenance

- Protocol/code commit before responses: `35fe9a8`
- Source model SHA-256:
  `473cf5c883929a2cf8b6d862bebf69e1bb6b6401bea8c7b0edba955e847b7e1d`
- Fresh model SHA-256:
  `6dfe230678f37ad4306ae77e7fbfda9c984a0e97b84f35b3742bff4f124e01d1`
- Policy SHA-256:
  `d1948b48b3f5307850a5505ce52b54f1c24fab94c09ecabc4f99fa1fc508c2b3`
- Public result SHA-256:
  `acc911ca350ba95ca221c7b676356374d49730fe0c71bcbfca889e96f761feef`
- Private raw-response SHA-256:
  `e04df8cee74cdb49ebc9c928afe242e80822173d12dadc5bcd1ee58b09d1e4f9`
- Authenticated OpenRouter balance after run: `$8.612217344`
- OatML/cluster use: none
