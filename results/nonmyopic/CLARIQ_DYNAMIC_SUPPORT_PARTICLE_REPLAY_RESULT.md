# ClariQ Dynamic-Support Joint-Particle Replay Result

## Decision

The zero-call particle replay fails the frozen first-link gates decisively.
There is no fresh V3 mechanics run, no development topic `148`, and no holdout
run. The ClariQ dynamic-support route closes.

This is a post hoc diagnostic on a failed serving tree, not a valid efficacy
result.

## Mechanics

The joint identity resolves the sole text collision without changing any
response:

| Metric | Result |
|---|---:|
| Parsed supports | `91 / 91` |
| Text-only collisions | `1` |
| Exact joint-particle collisions | `0` |
| Changed branch supports | `86 / 90` |
| Profile-diverse branches | `90 / 90` |
| Positive-continuation branches | `90 / 90` |
| Dynamic future-value range | `.702501` nats |

The generated supports are therefore path-dependent, non-collapsed, and
score-dynamic. The failure is not lack of branch variation.

## First-Link Failure

| Score | All-root Spearman with exact terminal NDCG | Selected root | Selected terminal NDCG |
|---|---:|---|---:|
| Myopic | `.487181` | `Q01056` | `.397611` |
| Fixed-support depth two | `.433013` | `Q00591` | `.359525` |
| Dynamic-support depth two | `.207143` | `Q02159` | `.320594` |
| Answer-link shuffled dynamic | `.410714` | `Q01056` | `.397611` |

Dynamic support loses `.077016` NDCG to myopic and shuffled dynamic, and
`.038931` to fixed-support depth two. It also has substantially worse global
ranking fidelity.

The shuffled control is especially diagnostic: retaining the same generated
branch values but breaking their answer links removes the harmful root change
and recovers the myopic winner. The model creates rich branch supports, but
their entropy values are not calibrated to the external retrieval value of the
answer-conditioned path.

## Interpretation

This reproduces the project’s recurring first-link bottleneck in a stronger
LLM-native setting:

1. path-dependent semantic beliefs exist;
2. their continuation EIG has large dynamic range;
3. but the mapping from a first answer to useful future semantic uncertainty is
   miscalibrated; and
4. non-myopic optimization amplifies that error.

Buying another topic, parser, or model under this exact construction is not
justified.

## Artifact and Budget

- Analysis:
  `results/nonmyopic/clariq_dynamic_support_particle_replay/ANALYSIS.json`
- SHA-256:
  `1dcabf28efbb3fc776fbddc56bf4ff7faea7dee2fbf1cc468f1a9309f2721699`
- New model calls / cost: `0 / $0`
- Project headroom remains: `$12.335724790776695`
- Development and holdout loaded: no
- OatML use: none
