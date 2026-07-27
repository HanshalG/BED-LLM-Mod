# DiscoverPhysics Dark-Matter Full-History Replay Result

## Decision

**The preregistered zero-call correction improved the dynamic policy but
failed the fixed-support gate. The mechanism diagnostic is a null.**

The result supports the diagnosis that two-branch quantization discarded
useful root information, but shows that this was not the whole reason the
LLM-refreshed support lost to fixed support.

## Frozen Correction

For refreshed hypothesis \(h\), branch representative \(c\), exact root
observation \(y_1\), continuation observation \(y_2\), and frozen
branch-conditioned weight \(w_h\), the replay used

\[
  p(h \mid y_1,y_2)
  \propto
  w_h
  \frac{p(y_1\mid h)}{p(c\mid h)}
  p(y_2\mid h).
\]

The exact failed-policy supports, actions, branches, observations, official
simulator, 96 hidden maps, common random numbers, and thresholds were reused.
There were no new model or API calls.

## Root Selection

| Root | Immediate EIG | Corrected internal trajectory risk |
|---|---:|---:|
| A | `1.1862` | `2.5366` |
| B | `1.0953` | `1.3516` |
| C | `.9049` | `2.0559` |
| D | `1.2177` | `2.8384` |

Myopic still selected northeast D and corrected lookahead still selected
center B. B reduced corrected internal risk by `52.4%` relative to D.

## Hidden Endpoint

| Policy | Official held-out trajectory MSE |
|---|---:|
| Corrected dynamic center B | `3.2490` |
| Corrected dynamic northeast D | `4.9956` |
| Corrected dynamic random northwest A | `4.4173` |
| Unchanged fixed-support center B | `2.9672` |

Corrected B beat D by `35.0%` and random A by `26.4%`. The frozen
stratified-bootstrap 95% interval for paired
`MSE(D)-MSE(B)` was `[1.302, 2.232]`.

However, corrected B remained `9.5%` worse than fixed-support B, failing the
sole unresolved gate. Relative to the original dynamic B MSE of `3.6161`,
the correction improved absolute MSE by `10.2%` and recovered `56.6%` of the
original dynamic-to-fixed gap.

## Post-Hoc Localization

The following decomposition is descriptive and was not a frozen gate:

| Hidden region | Corrected B | Fixed B | Corrected B minus fixed |
|---|---:|---:|---:|
| NE | `2.5065` | `2.8351` | `-.3285` |
| NW | `2.4202` | `1.5343` | `.8859` |
| SW | `6.5133` | `5.6635` | `.8497` |
| SE | `2.1766` | `2.4023` | `-.2257` |

Corrected dynamic support beats fixed support in NE and SE but loses in NW
and SW. Under the frozen asymmetric region prior, the residual weighted gap
is `.2817` MSE. A descriptive stratified bootstrap interval for
`MSE(fixed)-MSE(corrected)` is `[-.479, -.097]`.

This localizes two independent limitations:

1. Branch quantization discarded exact root information and materially hurt
   the original dynamic policy.
2. Even after restoring that information, the eight-hypothesis refreshed
   supports were less useful than the original support in the western
   regions.

The exact policy remains failed and is not repaired or rescued. Any fresh
confirmation needs a separately frozen protocol and independent model
outputs.

## Accounting

- New model/API calls: `0`
- OpenRouter cost: `$0`
- OatML use: none
- Result JSON:
  `results/nonmyopic/discoverphysics_dark_matter_full_history_replay.json`
- Result JSON SHA-256:
  `e82174d7af9ae971052ac85dc66775390f20e16baa232bc8a94858ade04c3c5f`
- Source policy SHA-256:
  `ab2b8a4ce3134fd12236e316b126cf00931f15cb78f818d00cca0b3a109abb46`
- Source frozen-model SHA-256:
  `7b13e0a3d105b3c823efe3f7bfe66dfb32f983058c9e0a279cd8f366c5a4b8e0`
