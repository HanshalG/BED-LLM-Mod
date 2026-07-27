# DiscoverPhysics Support-Gain Mechanism Analysis

## Status

**Post-hoc mechanism analysis on the already-confirmed 384-map endpoint.**

This analysis changes no policy, mixture weight, action, map, observation, or
gate and authorizes no new efficacy claim.

## Question

The retained-support policy improved over same-root fixed support by `2.41%`.
This could reflect useful new hypotheses, or the 5% component could act as an
unstructured regularizer.

For each confirmed map, this analysis measured:

- **coverage gain:** nearest held-out trajectory MSE under the initial support
  minus nearest MSE under the union of initial and routed regenerated support;
  and
- **endpoint gain:** fixed-support policy MSE minus retained-support policy
  MSE from the hash-bound confirmation.

The branch was assigned from the exact noiseless center-root response, matching
the confirmation's existing nearest-support diagnostic.

## Result

- Regenerated support improved nearest-hypothesis coverage on `274/384`
  maps (`71.35%`).
- Map-level Spearman correlation between coverage gain and endpoint gain:
  `.2794`.
- Mean endpoint gain when coverage improved: `+.1612` MSE.
- Mean endpoint gain when coverage did not improve: `-.1220` MSE.
- Endpoint win rate when coverage improved: `62.04%`.
- Endpoint win rate when coverage did not improve: `46.36%`.
- Prior-weighted coverage gain: `.3930` MSE.
- Prior-weighted endpoint gain: `.0722` MSE.

## Regional Heterogeneity

| Region | Coverage-help rate | Mean coverage gain | Mean endpoint gain | Spearman |
|---|---:|---:|---:|---:|
| NE | `61.5%` | `.4550` | `.2945` | `.6318` |
| NW | `39.6%` | `.1321` | `-.2416` | `-.1281` |
| SW | `84.4%` | `.5898` | `.0017` | `-.1343` |
| SE | `100%` | `.5339` | `.2657` | `.3830` |

The positive endpoint is driven by NE and SE, especially the high-prior NE
region. NW loses despite modest coverage expansion, and SW coverage expansion
does not translate into policy gain. This explains why low regenerated mass
worked while equal weighting and replacement did not.

## Interpretation

The aggregate association is consistent with genuine hypothesis-space
expansion: maps receiving a closer regenerated hypothesis are more likely to
benefit from the retained policy. It is not proof of causality. Nearest-support
MSE is a geometric proxy, support generation and region are confounded, and
the analysis was specified after seeing the positive endpoint.

The strongest defensible chain is therefore:

1. the fresh-map retained-support policy passes all preregistered efficacy
   gates;
2. its same-root gain cannot be attributed only to selecting central root B;
3. post hoc, that gain is directionally linked to where regenerated support
   expands physical coverage; and
4. a fresh-tree robustness attempt fails serving before policy evaluation.

## Accounting

- New model/API calls: `0`
- OpenRouter cost: `$0`
- OatML use: none
- Analysis JSON:
  `results/nonmyopic/discoverphysics_dark_matter_support_gain_mechanism.json`
- Analysis JSON SHA-256:
  `4bad9429d90a5f6f9d4fdbf04f49296a79939196e49263a9fc25df1ef3ae002b`
- Source confirmation SHA-256:
  `aecc11fc2b5efc1e7755137d727a58a7fd69d107ca19035b55d38015145050b9`
