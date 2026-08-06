# Number Game August 7 Precharge Reservation Amendment

Date frozen: 2026-08-06

## Problem

The OpenRouter spend ledger serialized completed-response writes, but each
request was charged before its cost entered that transaction. With concurrent
requests, multiple in-flight responses could therefore consume the same
remaining run allowance. A post-response exception would stop the run but
could not undo the provider charge.

## Safety Amendment

Before each frozen Number Game HTTP attempt, atomically reserve a conservative
maximum attempt cost in the shared spend ledger. Completed responses settle
actual provider-reported cost against that reservation. Concurrent workers and
retries wait for settled headroom rather than dispatching beyond the run cap.

Reservations are shared across threads, adapters, and processes through the
existing file lock. A missing usage cost, transport ambiguity, charge above the
reserved maximum, or process interruption leaves that attempt's reservation in
place and fails closed until account-level reconciliation. Any retry must then
acquire a separate reservation. Explicit HTTP errors and zero-cost provider
error responses release their reservation before the registered retry because
they did not return a billable completion.

Frozen per-request maxima:

| Model | Maximum request cost |
| --- | ---: |
| `qwen/qwen3.7-plus` | `$0.0100` |
| `openai/gpt-5.6-luna` | `$0.0040` |
| `deepseek/deepseek-v4-flash-0731` | `$0.0015` |

The August 7 preflight checks these exact values against the live catalog's
input/output prices and each model's 4,200-token output ceiling. It requires
the residual reservation to cover at least 8,000 prompt tokens at the live
input price.

## Scientific Boundary

This changes only precharge budget enforcement. Models, prompts, response
schemas, reasoning settings, temperatures, seeds, case inventories, retry
rules, parsers, support thresholds, endpoints, and scientific gates are
unchanged. No model response or endpoint was opened before this amendment.
