# Number Game Diversity Confirmation Precharge Amendment

Date frozen: 2026-08-06, before the August 7 predecessor control and before
either prospective August 8--9 confirmation block.

## Problem

The prospective diversity confirmation uses Qwen 3.7 Plus for path-dependent
belief generation and Gemini 2.5 Flash for target and validation-support
generation. The shared precharge guard covered Qwen but did not yet assign a
worst-case attempt reservation to Gemini. Gemini requests could therefore be
dispatched concurrently against the same apparent remaining run allowance.

## Safety Amendment

Every HTTP attempt from either frozen model must reserve worst-case cost under
the shared spend-ledger lock before dispatch:

| Model | Maximum attempt cost |
| --- | ---: |
| `qwen/qwen3.7-plus` | `$0.0100` |
| `google/gemini-2.5-flash` | `$0.0150` |

At the catalog prices observed when this amendment was frozen, each maximum
covers the 4,200-token output ceiling and at least 8,000 prompt tokens. The
formal daily preflight must recheck exact endpoint availability, pricing,
output capacity, reservation values, and prompt-token coverage before opening
either block. Ambiguous transport failures retain their attempt reservation;
retries require separate headroom.

The source adapter's existing configured override remains `$5.00`, equal to
the account-wide London-day cap. This amendment adds transport-level
enforcement; it does not increase any allowance.

## Scientific Boundary

This changes only precharge budget enforcement. Models, prompts, structured
schemas, reasoning settings, temperatures, seeds, case inventory, retry rules,
parsers, support construction, selectors, baselines, endpoints, and frozen
scientific gates are unchanged. No prospective model response, tree, target,
or endpoint was opened before this amendment.
