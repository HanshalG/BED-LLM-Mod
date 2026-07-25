# ClariQ Dynamic-Support V1 Smoke Result

## Decision

V1 fails closed on the first initial-support response. No branch support,
policy score, selected root, or NDCG endpoint exists. Topic `148` and the four
holdout topics remain untouched.

This is a transport failure, not evidence for or against path-dependent dynamic
support.

## Failure

The frozen grammar required a contiguous 13-character response-code string.
GPT-5.4 returned all eight ordered hypothesis lines, and every line contained
exactly 13 valid `A/B/C` codes, but it separated those codes with single spaces.
The raw field therefore had length 25 and strict parsing stopped immediately.

The response ended normally (`finish_reason=stop`) after 259 completion tokens;
there was no truncation, reasoning, forced exit, retry, malformed mass, missing
hypothesis, or missing code.

Per registration:

- the response is not cleaned or reused;
- no branch requests are sent;
- topic `38` is not rerun under V1;
- no score or endpoint is reconstructed; and
- development and holdout remain sealed.

## Usage and Artifacts

- Requests / HTTP attempts: `1 / 1`
- Prompt / completion tokens: `1,303 / 259`
- Retries / reasoning / forced exits: `0 / 0 / 0`
- Cost: `$0.0071425`
- Public failure:
  `results/nonmyopic/clariq_dynamic_support_smoke/clariq-dynamic-support-smoke-20260725T234425Z/SMOKE_FAILURE.json`
- Public SHA-256:
  `5175770604852793c5d97e4a8c9072a2261206a70ae2e6139e73e2f3e2d2a844`
- Private raw SHA-256:
  `07735493d6b72895fa0dabde789be4b02601af303084bb35e7eff940e1b5f3d2`

The project ledger is `$91.90817570922331` spent with
`$13.091824290776685` remaining under its stricter cap. The authenticated
provider endpoint still reports `$38.483769494` remaining, so the protected
`$25` Monday reserve is intact. OatML use: none.
