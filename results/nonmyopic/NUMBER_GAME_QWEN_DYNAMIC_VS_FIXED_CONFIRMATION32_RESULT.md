# Number Game Qwen Dynamic-vs-Fixed Confirmation32 Result

Date: 2026-07-29

## Status

**Gated null.** The fresh mean Brier effect favors path-dependent dynamic
support, but the preregistered confidence-interval, win-count, and one
mechanics gate fail.

## Frozen Primary

Across 32 fresh pooled-Qwen trees, each scored with 16 independent Gemini
validation supports and the exact 33-concept canonical endpoint:

| Metric | Dynamic support d3 | Fixed initial-support d3 | Difference |
|---|---:|---:|---:|
| Brier | 0.0996438 | 0.1042844 | -0.0046406 |
| Hamming | 0.0210584 | 0.0223147 | -0.0012564 |

The Brier reduction is 4.45%, above the preregistered 3% magnitude gate.
Dynamic and fixed roots differ on 25/32 trees. However:

- paired 95% Brier-difference interval:
  `[-0.0115988, +0.0003716]`;
- Brier wins/ties/losses: `14/7/11`;
- required wins: at least `16`.

The root-difference and mean-magnitude gates pass. The confidence-interval and
win-count gates fail.

## Co-Required Policy Result

Path-dependent depth three remains strongly better than myopic EIG:

- Brier: `0.0996438` versus `0.1203313`;
- relative reduction: `17.19%`;
- paired 95% Brier-difference interval:
  `[-0.0283998, -0.0135837]`;
- Brier wins: `26/32`;
- Hamming reduction: `23.48%`;
- mean canonical coverage improvement: `+4.26` percentage points.

All three preregistered depth-three-versus-myopic gates pass.

## Mechanics

- exactly `32` fresh trees;
- exactly `3680` accepted requests;
- `3697` HTTP attempts and `17` retries;
- one provider-error retry, within the frozen cap;
- `3679/3680` strict JSON draws and one item-salvaged draw, within cap;
- zero reasoning tokens and zero forced exits;
- cost: `$4.6478466`, below the `$5.25` cap;
- all pooled initial supports and all 16 validation supports per tree pass;
- one deployed merged first branch on tree 16 has `11` hypotheses versus the
  preregistered minimum `12`.

That single support minimum makes the all-retained-branches mechanics gate
false. It is reported without relaxation or replacement.

## Diagnostic Only

The largest favorable tree has a Brier difference of `-0.0935116` and
accounts for 62.97% of the summed improvement. Removing only that tree changes
the mean difference from `-0.0046406` to `-0.0017738`.

This leave-one-tree diagnostic was not preregistered and cannot alter status.
It reinforces the registered conclusion: the fresh result is directionally
suggestive but does not confirm a stable causal regeneration increment.

## Interpretation

The earlier retrospective 64-tree dynamic-versus-fixed result remains a
retrospective 3.79% association. The fresh experiment reproduces its direction
and clears the magnitude threshold, but it does not clear uncertainty or
win-count gates. Therefore:

- non-myopic depth-three policy efficacy over myopic selection is robust;
- the causal value of LLM path-dependent support regeneration remains
  unconfirmed;
- no rerun, subset, outlier exclusion, threshold repair, or mechanism rescue
  is authorized.

## Artifact Hashes

- RESULT:
  `75f55d7ffe792771de8d2dd909c32208a71b8dd169a171b8148e958e6eb0458e`
- TREES:
  `dff467d9590854a52c80ae0fec4d3a9ef3cedb315a8712258fd25bed1dce2149`
- TARGETS:
  `f4113ebac996b625ff5cce3e436357e940096e46311c7c68bc8b9acb28b54e5f`
- Private raw responses:
  `112ebca54cd8b3e0a91adda4b97c60a5cb68da51f1fa3c73d9b204e0854c6997`
