# Bongard Endpoint-Utility Readiness

Checked 2026-08-08 16:59 London time against pushed commit
`9aa76aff26a2adfb28ffb57768362d025118c41e`.

Status: **ready without paid calls** for the frozen August 10 serving and
mechanics sequence.

## Bound Protocol

- August 10 wrapper V4: `8a020e16532f25d7f2b98398dbf3c82d124c525b6227d3ee0adf56ca93d9cd04`
- Endpoint-predictive amendment: `2fce4b66696d5b635f8d32c5f968a917aac20ab64305cab2728bc5f9973a55b8`
- Development64 V14: `377596232d9fda34753bd99914292043ecc80a5584d55d95075e659c40e88011`
- Naive first-link V4: `223f6c75cb6db0f75d5557bb146c8e88a5a3d2756c7c6211b21c8a7b88df04ec`
- Confirmation96 V11: `42e965d6ee66d36f2a0eac38b5cfc8b18567e16684f2dce7c5fd7e8daff797f1`

The complete Bongard suite passes 145/145. The exact hypothesis-EIG
counterexample, endpoint-label invariance, signed shuffled continuation,
manifest replay, and unchanged request-count tests all pass.

## Live Preflight

The production `--preflight` command made zero model calls and wrote zero
files. Wrapper, serving, mechanics, and August 10 ledger paths are all absent.
The live Luna endpoint still accepts image and text input, supports structured
output, and exposes 1,050,000 context and 128,000 completion tokens at
`$0.10/M` input and `$0.60/M` output.

Authenticated OpenRouter totals were credits `$245.00`, usage
`$220.121013787`, and balance `$24.878986213`. The frozen serving and mechanics
caps total `$2.00` under the hard account-wide `$5` London-day cap.

No candidate label, endpoint label, Bongard model response, or scientific
endpoint was accessed. On August 10, run a fresh same-day preflight and invoke
the exact wrapper once only when it still reports ready.
