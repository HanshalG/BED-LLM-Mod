# Bongard Luna Precharge Amendment

Date frozen: 2026-08-06, before the August 10 serving/mechanics sequence and
before every August 11--14 development block.

## Problem

The Bongard serving, mechanics, and development adapters issue concurrent
multimodal Luna requests. Their run-level budgets were checked after provider
responses, but no worst-case cost was reserved before dispatch. Multiple
in-flight attempts could therefore consume the same apparent remaining run
allowance before their charges reached the shared ledger.

## Safety Amendment

Every `openai/gpt-5.6-luna` HTTP attempt reserves `$0.0040` under the shared
spend-ledger lock before dispatch. At the catalog prices observed when this
amendment was frozen, that covers the 3,200-token output ceiling plus 20,800
prompt tokens. Every August 10--14 runtime preflight must recheck the exact
endpoint, text/image and structured-output support, live prices, output
capacity, reservation value, and at least 8,000 prompt tokens of residual
coverage.

Ambiguous transport failures retain their attempt reservation and retries
must acquire separate headroom. Explicit HTTP errors and zero-cost provider
errors release their reservations before retry. Existing component run caps
of `$0.25`, `$1.75`, and `$4.75` remain unchanged and cannot be exceeded by
concurrent dispatch.

## Scientific Boundary

This changes only transport-level budget enforcement. The model endpoint,
images, prompts, schemas, reasoning setting, temperatures, seeds, batching,
task inventory, hypothesis representation, likelihoods, policies, baselines,
endpoints, and scientific gates are unchanged. No prospective Bongard model
response or endpoint was opened before this amendment.
