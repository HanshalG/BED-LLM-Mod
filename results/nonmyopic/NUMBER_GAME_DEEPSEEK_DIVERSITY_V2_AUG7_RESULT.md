# Number Game DeepSeek Diversity V2 Aug 7 Result

Date: 2026-08-07 (Europe/London)

## Result

The fresh DeepSeek V4 Flash 0731 diversity-V2 reliability gate is a clean
mechanics null. It made exactly 128 accepted requests with no retries, provider
errors, reasoning tokens, forced exits, or strict-JSON failures and cost
`$0.03368175`.

All eight initial supports had 22--24 valid extensions. Across 120 fresh
conditioned histories, valid-support minimum/mean/maximum was `0 / 8.625 / 20`.
Eighteen draws fell below the frozen floor of four, including one zero-support
draw, so `all_conditioned_supports_have_at_least_4_valid` failed. Stress7168
was not authorized and made zero calls.

## Mechanism

Relative to the disjoint unchanged-interface DeepSeek gate, V2 removed most
extension duplication and schema drift:

| Diagnostic | Original | Diversity V2 |
|---|---:|---:|
| Duplicate-extension rejections | 820 | 33 |
| Wrong-field rejections | 48 | 0 |
| Invalid-expression rejections | 68 | 18 |
| Observation-inconsistent rejections | 507 | 1,800 |
| Conditioned valid mean | 12.108 | 8.625 |
| Conditioned draws below 4 | 20 / 120 | 18 / 120 |
| Zero-support draws | 7 / 120 | 1 / 120 |

The six-family instruction diversified extensions, but the nonreasoning model
could not simultaneously enforce two observed labels across those families.
This is not a near-pass: average support quality worsened and the long tail
remained. Close nonreasoning DeepSeek prompt-only development for this Number
Game interface. A future route would need a substantively different mechanism,
such as pooled multisampling or reasoning, and must use new cases and a new
preregistration.

## Spend And Boundaries

Aug 7 recorded spend is now `$2.815715891`, leaving `$2.184284109` under the
`$5` cap. The unused stress allocation is deliberately unspent. No policy
efficacy, selected action, truth, or Qwen endpoint was accessed, and this result
does not authorize Aug 8 diversity.

## Artifact Bindings

- wrapper RESULT:
  `4e87b8d823cf5dee0bdb52b7c1438ca42878c2098453abe6f87349797c8a2890`;
- reliability RESULT:
  `cd451acd04e2c98a6760267a4b6cb4f66fdb94b3a88f5f3fea10625b0d1ec788`;
- private raw responses:
  `97685ad1140384c12bbb94470a67dd0487097631b44838d0d2313637b106d5ca`;
- zero-call stress RESULT:
  `3350ff6ba1c215e37d3cd96ad8273b88135512115dc72d2a263467a7c69439f3`;
- reconciled Aug 7 ledger:
  `240fca9ad7cc3de1c732b804ce7fe90818ffec6a524ec34e667a452edf7ce423`.
