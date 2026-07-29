# Human Semantic Norms Object Game: Opportunity Result

Date executed: 2026-07-29

Status: **the frozen zero-call opportunity gate failed; no LLM serving is
authorized for this construction.**

## Integrity

- Preregistration commit: `e24dd97`
- Source commit:
  `d2b940472bb9e88983a5c087594022939cac8895`
- Public result SHA-256:
  `cf4971ef2b83938babae3c0172ad48cb253b4eb44f78a28564a721d3cf1fb146`
- Model calls: `0`
- Cost: `$0`

All four pinned source hashes matched. The deterministic selection produced
exactly 32 objects from eight categories, with object-order SHA-256
`3efb009e03a2c6139dcf31cfe4b3aa2bf5a26bbd1b15bcc078a4be5628f842df`.

## Source Quality

The semantic endpoint itself was healthy:

| Quantity | Result | Gate |
|---|---:|---:|
| eligible English objects | 199 | >=96 |
| selected objects/categories | 32 / 8 | 32 / 8 |
| valid human feature columns | 409 | descriptive |
| distinct extensions | 392 | >=64 |
| 8--24-member extensions | 80 | >=16 |
| support-size min / median / max | 3 / 4 / 29 | descriptive |

The extension-bank SHA-256 is
`27a96cdae5517209f89cf387bdb32f0dfc20476cab2ad7de8a4546eb524a8a8c`.
Raw production vectors and Finnish feature strings remain external and are not
redistributed.

## Planning Result

| Horizon | Exact root | Greedy root | Exact entropy | Greedy entropy | Entropy gain | Brier gain |
|---:|---|---|---:|---:|---:|---:|
| 1 | finger | finger | 5.420417 | 5.420417 | 0 | 0 |
| 2 | finger | finger | 4.858961 | 4.858961 | 0 | 0 |
| 3 | finger | finger | 4.282432 | 4.282432 | 0 | 0 |

The three scientific gates all failed: the depth-three root did not change,
entropy gain was below `0.005` nats, and Brier gain was below `0.001`.

## Interpretation

This is not a serving or model failure. Human feature production creates a
large, natural, independently authored hypothesis bank, but the permitted
object-membership questions make balanced immediate splits also optimal for
the full three-question tree. The setting lacks the first-link
immediate-information sacrifice required for non-myopic BED.

Per the preregistration, there will be no alternate object-selection seed,
category subset, feature weighting, membership threshold, or paid LLM smoke.
The exact construction is closed.
