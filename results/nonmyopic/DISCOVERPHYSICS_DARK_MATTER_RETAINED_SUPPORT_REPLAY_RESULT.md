# DiscoverPhysics Dark-Matter Retained-Support Replay Result

## Decision

**The preregistered retained-support replay passed every gate except the
fixed-support improvement gate. It is a diagnostic null.**

Retaining the original hypotheses alongside branch-regenerated hypotheses
removed most of the dynamic policy's deficit, but did not make support
regeneration improve over the same-root fixed-support policy.

## Frozen Method

At each branch representative, the replay assigned:

- mass `.5` to the exact branch posterior over the eight original
  hypotheses; and
- mass `.5` to the frozen LLM weights over the eight regenerated
  hypotheses.

All 16 particles were then updated using official-simulator likelihood
ratios for the exact root observation and the continuation. No support was
deduplicated, no mixture weight was tuned, and no model/API calls were made.

## Root Selection

| Root | Immediate EIG | Retained-support internal risk |
|---|---:|---:|
| A | `1.1862` | `.9877` |
| B | `1.0953` | `.7601` |
| C | `.9049` | `1.8369` |
| D | `1.2177` | `2.4494` |

Myopic selected northeast D and retained-support lookahead selected center B.
B reduced internal risk by `69.0%` relative to D.

## Hidden Endpoint

| Policy | Official held-out trajectory MSE |
|---|---:|
| Retained-support center B | `3.0777` |
| Retained-support northeast D | `4.6882` |
| Retained-support random northwest A | `3.6100` |
| Unchanged fixed-support center B | `2.9672` |

Retained B beat D by `34.4%` and random A by `14.7%`. The frozen
stratified-bootstrap 95% interval for paired `MSE(D)-MSE(B)` was
`[1.175, 2.102]`.

Retained B nevertheless remained `3.7%` worse than fixed-support B, so it
failed the frozen requirement of at least a `5%` improvement. Relative to
the original dynamic B MSE of `3.6161`, retention improved absolute MSE by
`14.9%` and recovered `83.0%` of the original dynamic-to-fixed gap.

Nearest-support trajectory risk improved from `1.7311` to `1.3708`, a
`20.8%` reduction.

## Post-Hoc Localization

The following decomposition is descriptive and was not a frozen gate:

| Hidden region | Retained B | Fixed B | Retained B minus fixed |
|---|---:|---:|---:|
| NE | `2.4151` | `2.8351` | `-.4200` |
| NW | `2.2547` | `1.5343` | `.7204` |
| SW | `6.0923` | `5.6635` | `.4288` |
| SE | `2.1682` | `2.4023` | `-.2341` |

Retained B wins 45 of 96 maps and improves on fixed support in NE and SE,
but still loses in NW and SW. The weighted difference
`MSE(fixed)-MSE(retained)` is `-.1105`; its descriptive stratified-bootstrap
95% interval is `[-.258, .030]`.

Support replacement was therefore a major architecture error, but retention
only brings dynamic support near parity. The exact policy remains failed and
is not rescued. Any mixture-weight exploration on these maps is post hoc and
cannot support a claim; fresh confirmation would require an independently
frozen protocol, outputs, and endpoint.

## Accounting

- New model/API calls: `0`
- OpenRouter cost: `$0`
- OatML use: none
- Result JSON:
  `results/nonmyopic/discoverphysics_dark_matter_retained_support_replay.json`
- Result JSON SHA-256:
  `e5c1812f85bd0301db7346ac0929bec299e65f828d876ccc0af8379658f4e1da`
- Source policy SHA-256:
  `ab2b8a4ce3134fd12236e316b126cf00931f15cb78f818d00cca0b3a109abb46`
- Source frozen-model SHA-256:
  `7b13e0a3d105b3c823efe3f7bfe66dfb32f983058c9e0a279cd8f366c5a4b8e0`
