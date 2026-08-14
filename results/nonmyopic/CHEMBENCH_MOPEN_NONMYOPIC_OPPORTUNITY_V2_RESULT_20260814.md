# ChemBench M-open Non-Myopic Opportunity V2 Result

Date: 2026-08-14 (Europe/London)

## Decision

The frozen zero-call structural gate passed. On all 399 paired truth cells from
the seven untouched ChemBench validation slices, expected terminal log-rate MSE
decreased strictly from depth one to depth two to depth three. The improvement
was 17.35% for d2 versus d1 and 26.40% for d3 versus d2.

This result authorizes implementation of the zero-call residual-conditioned
M-open mechanics only. It is not evidence that an LLM can propose useful
mechanisms, that a learned policy beats a control, or that the eventual method
is publishable.

## Immutable Bindings

- Official source repository: `scientific-discovery/LLM-AutoSciLab`
- Source commit: `acf160eb6c96897748dd92b152703b59b74efc05`
- Source tree: `e042d418fc30c6c70f6d1c6b0636d43f0c1c0f7a`
- Protocol SHA256: `42f9f79edd0372daa045ef627909c77328c7727a83d337773e27dfc2e79fadce`
- Implementation SHA256: `2f13322cb7e4552f9fd0d0e0b86e1436014c136716d44abfa58b6b67fe76336b`
- Focused test SHA256: `36f51f33f3b0d4cefd1ff626cd2fbbe0e036363efe92f596eda8e80088d3fbe6`
- Result SHA256: `54309e271600d2d9b5f848c797217c568735d12a0efb8bef6643998fa6b34edd`
- Active-domain list SHA256: `38d0c72e5a4d828fec072e804f664c010e766416dfe11c4585b2f03245dedb18`
- Model/API calls: `0`
- Cost: `$0`

## Aggregate Result

| Horizon | Expected terminal MSE | Expected terminal RMSLE |
| --- | ---: | ---: |
| d1 | 0.03672927 | 0.19164882 |
| d2 | 0.03035790 | 0.17423519 |
| d3 | 0.02234274 | 0.14947488 |

| Comparison | Relative MSE reduction | Truth-cell wins / ties / losses |
| --- | ---: | ---: |
| d2 vs d1 | 17.35% | 135 / 196 / 68 |
| d3 vs d2 | 26.40% | 111 / 204 / 84 |
| d3 vs d1 | 39.17% | 155 / 186 / 58 |

All seven conjunctive gates passed: both successive aggregate reductions
exceeded 5%, d2 beat d1 on six of seven slices, d3 beat d2 on all seven, d3
beat d1 on all seven, and each successive comparison won more truth cells than
it lost.

## Slice Results

| Slice | d1 MSE | d2 MSE | d3 MSE | d2 vs d1 | d3 vs d2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| easy/v2 | 0.034924 | 0.025977 | 0.020367 | 25.62% | 21.60% |
| medium/v0 | 0.017952 | 0.007012 | 0.002646 | 60.94% | 62.26% |
| medium/v1 | 0.028787 | 0.017251 | 0.016418 | 40.07% | 4.83% |
| medium/v2 | 0.029301 | 0.025714 | 0.022644 | 12.24% | 11.94% |
| hard/v0 | 0.061677 | 0.055131 | 0.048999 | 10.61% | 11.12% |
| hard/v1 | 0.038085 | 0.042757 | 0.026115 | -12.27% | 38.92% |
| hard/v2 | 0.046379 | 0.038664 | 0.019210 | 16.64% | 50.32% |

The selected root assay changed between d1 and d2 on five of seven slices and
between d2 and d3 on four of seven. This is a genuine planning effect rather
than deeper evaluation of the same first action.

## Interpretation

ChemBench is a better environment for the MDA-derived research direction than
stock NeuronBench. It has a large compound-mechanism space, an explicit M-open
failure mode, seven-dimensional experiments, and a prediction metric that can
reward discovering the right executable structure. The result establishes
that the experiment geometry contains enough horizon opportunity to justify an
LLM-native test.

The audit is intentionally favorable to the planner: all 57 released active
mechanisms are present from the start, each mechanism uses one fixed released
parameter point per slice, and observations are reduced to three bins. The
next mechanics stage must remove the complete-support advantage. It must start
from a restricted live support, expose residuals to an executable-model
proposer, and measure whether branch-conditioned support expansion preserves
the depth advantage against call-matched controls.
