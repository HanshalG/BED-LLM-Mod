# ClariQ Dynamic-Support V2 Mechanics Result

## Decision

The full V2 mechanics tree fails closed before policy scoring or endpoint
loading. Dynamic-support first-link efficacy is unmeasured. Development topic
`148` and holdouts `102/11/103/141` remain untouched.

## Failure

All 91 requested responses arrived normally:

- one fresh initial support and 90 branch supports;
- 91 physical requests and HTTP attempts;
- zero retries, reasoning tokens, or forced exits;
- every response ended with `finish_reason=stop`; and
- all response-code fields had the exact V2 spaced shape.

The initial support and 89/90 branch supports passed strict parsing. In branch
`Q00628:B`, H07 and H08 both described the normalized intent “homepage for
Bellevue Hospital Center in New York City,” with mass `1` each, while assigning
different 14-code future-response profiles. The frozen parser required eight
distinct normalized intent texts and rejected this branch.

Per registration, there is no deduplication, merge, particle reinterpretation,
partial-tree score, subset analysis, response repair/reissue, or V2 rerun. No
NDCG value was loaded.

## Interpretation

V2 solved the response transport problem and was nearly complete, but it
exposed a modeling ambiguity: is a support element an intent string alone, or a
joint particle consisting of intent text and a stochastic response profile?
The failed run cannot answer that question. A zero-call, separately frozen
diagnostic may evaluate the latter interpretation on this disclosed tree before
any fresh API spend; it cannot turn V2 into a valid result.

## Usage and Artifacts

- Prompt / completion tokens: `216,655 / 20,119`
- Cost: `$0.7466545`
- Public failure:
  `results/nonmyopic/clariq_dynamic_support_v2_mechanics/clariq-dynamic-support-v2-mechanics-20260725T235419Z/MECHANICS_FAILURE.json`
- Public SHA-256:
  `8a29b3e4960453610ab186612e92411bca14f3b71b037a6d6d9fbfcb12f1cefb`
- Private raw SHA-256:
  `752718c675f3e2c9d7f89dcb94b2a26286e563c9e68a1ea0c8237d8ff78bab9e`
- Project spend / headroom:
  `$92.6642752092233 / $12.335724790776695`
- Live remaining / above reserve:
  `$37.720527494 / $12.720527494`
- OatML use: none
