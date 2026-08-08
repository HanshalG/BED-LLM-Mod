# Bongard OpenWorld Luna Transport-Retry Audit

Date: 2026-08-08. Model calls: `0`. Cost: `$0.00`.

## Finding

The unopened Bongard serving, mechanics, development, and confirmation gates
required exact equality between accepted responses and HTTP attempts and also
required zero retries. The Aug 8 RegretBench transaction demonstrated that
this makes a transient response-delivery failure capable of vetoing a complete
scientific pathway independently of semantic quality.

The gate was stricter than the estimand. OpenRouter's adapter retries the same
serialized request, seed, schema, and decoding configuration when transport
fails before a usable response is available. Every attempt acquires a separate
worst-case cost reservation. A zero-cost provider-error response is likewise
retried without content-based selection.

## Repair

The prospective amendment replaces the infrastructure-only zero-retry veto
with one shared gate used by all four unopened stages. It requires:

- exact preregistered accepted-response count;
- non-negative integer usage counts;
- `http_attempts == accepted_responses + retry_count`;
- no more than `max(4, ceil(0.02 * accepted_responses))` retries;
- provider-error retries to be a non-negative subset of total retries.

Strict schemas, exact semantic support, branch and terminal label obedience,
paired seeds, task-level CRN, privacy, endpoint sealing, finite scores, all
policy controls, efficacy thresholds, run caps, and the account-wide daily cap
are unchanged. A request set that never reaches the exact accepted-response
count still fails closed and cannot resume in place.

The rebound development manifest distinguishes `344` accepted responses from
`351` maximum HTTP attempts per block. Confirmation distinguishes `688` from
`702`; its maximum precharged exposure is therefore `$2.808`, still below both
frozen caps.

## Adversarial Verification

The exact-10 integration fixture passes with one correctly accounted retry.
The shared gate accepts the frozen four-retry minimum boundary and rejects:

- five retries on an exact-10 stage;
- an HTTP-attempt count inconsistent with accepted responses plus retries;
- provider-error retries exceeding total retries;
- fractional usage counts;
- eight retries on a 344-response development block, where the exact bound is
  seven.

The complete Bongard test family passes:

```text
133 passed in 59.52s
```

## Authoritative Bindings

| Artifact | SHA-256 |
|---|---|
| Retry amendment | `0f6ffc66f9b7d0f45d8913cf4f790d6f0c891e135cf5b9102575bd0a51ef7dd9` |
| Serving interface V4 | `84642d65895e2593b7cb7a63fbe818d9f0695dbab1fa767d90b14cddf388a610` |
| Mechanics interface V10 | `a499fe8fa11888064288bde1ce92d5f376424ff5e15bdd0a3c8d4b02118f7d29` |
| Development interface V12 | `5f4a88a967dd2b336e17a3d66ba8ead1a09cb9156cddbf857a6ca4f89c54604d` |
| Development V12 manifest | `3e52e97c1ff28968273bedeea37ca2695cb41df5aff6478a3c3848b1bbee2ae0` |
| Naive block manifest V2 | `2c6f3de36be214d5eab9f84519d0a86be4780bdffa839eedd6cc0ff4a3e3ab9f` |
| Confirmation interface V6 | `ae0d15118b2e0adcce8cbac0372649a39a7b036e737d7380cc759104e646af49` |
| Confirmation manifest V8 | `94c81d778f7cf939912ce0253883527f1494431dcf1fb7908514adb775d7b229` |
| Confirmation daily executor V2 | `cfae151f8a54f6bc007d077546e675be3e09d101371eeb7a0cf5353c372809a8` |
| Confirmation execution verifier V3 | `e81cc2b41c729307c26cbed2700fc311fd45135f593824db4888d7cbee5c23cc` |
| Aug 10 wrapper | `4592662c24242eb17e9de004c080a25f624bb9ffe6afd4aa8151cc73a5415bea` |

The earlier development V11 and confirmation V7 manifests remain historical
and cannot authorize future execution. The already banked Aug 8 Luna naive
smoke remains unchanged and independently verifiable; V2 governs only the
future block-to-development binding.

## Readiness

The exact Aug 10 CLI preflight returns `ready_without_paid_calls`, validates the
live Luna multimodal/structured-output contract, binds development manifest
`3e52e97c...`, sees all wrapper/serving/mechanics/ledger paths absent, and makes
zero model calls or writes. The frozen component cap remains `$2.00`.
