# Public source journal: synthetic failure coverage

Previous goal turn was progress: real source collection failed before any paid
call, exposing lost failure metadata. This turn implements a one-shot public
schedule journal without reopening that cohort or generating new outcomes.

Each exact task/seed/channel request is saved and flushed before dispatch.
Successes preserve only validated public channels with hashes. Failures record
the exact schedule index/request, phase, exception type, return code when
available, stdout/stderr byte counts and hashes, and timeout bound when relevant.
Arbitrary exception text and raw failure stdout/stderr are not persisted because
they may contain withheld labels. This distinguishes transport exits from channel
validation failures; it does not yet reveal the underlying generator/verifier
exception. No fabricated retrospective diagnosis of the old failure is made.

The collector refuses duplicate examples, hidden-output mode and existing output
directories. A failure preserves the successful prefix and stops without retry.
Prefix replay checks schedule coverage, channel validity, request/receipt identity,
public content hashes and unexpected files. An externally frozen bank hash is still
needed for adversarial whole-bank integrity; this helper is not a cryptographic
attestation of the worker or an end-to-end qualification runner.

Nine synthetic tests cover nonzero exit, oversized stdout, accidental output
exposure, timeout, malformed and duplicate-key JSON, success, repeat refusal,
forbidden schedule and modified public content. No benchmark/source/model calls.

Next integrate this helper with the existing bounded Docker transport and a
structured trusted-worker failure envelope, then prospectively freeze a fresh
exact public schedule and bank it once. Paid gates, task selection and scientific
thresholds remain unchanged; no repaired source artifact can rescue the closed
paired-repair cohort. Research goal remains unachieved. API balance and daily
allowance unchanged; no cluster or automation changes.
