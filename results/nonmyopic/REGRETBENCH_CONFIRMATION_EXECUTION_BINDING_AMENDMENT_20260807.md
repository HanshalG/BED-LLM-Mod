# RegretBench Confirmation Execution-Binding Amendment

Date frozen: 2026-08-07

## Purpose

Bind the producer and independent result verifier required by the already
frozen RegretBench dynamic depth-two confirmation protocol. This amendment is
made before any support, policy-development, or confirmation model response.

It changes no task, model, prompt, support width, policy, endpoint, matcher,
penalty, seed, request count, budget, mechanics threshold, science threshold,
or authorization rule in the confirmation preregistration or protocol
manifest.

## Frozen Parent

- confirmation preregistration:
  `edbbe7cd9e7ef582fc74629737e6c9261cead4486800bfc8712b0c74e90dd8bc`;
- confirmation protocol manifest:
  `7a782f02eb8c3b16d5b229cca309d02bce64df6432c5090c977d3e26d1f46498`;
- original zero-call protocol verification:
  `20a47fd4cde521327a1ea1c3c183a6efad0cd0b580aed6773e49829fd841ab3c`.

The parent remains the scientific authority.

## Mechanical Verifier Generalization

The independent RegretBench result verifier originally embedded the
development initial-seed start `202608089000` as a literal while every other
schedule component was a named constant. Before responses, that literal is
replaced by `INITIAL_SEED_START`, whose development default remains exactly
`202608089000`.

- old verifier SHA-256:
  `dd67501dd311cc5882c0274289aeab95b7d3bcb57af8bc7f0d5b79b450957e8f`;
- generalized verifier SHA-256:
  `ad5f516b473cddd4b4957f4921752c85e56936e28517fff1a10a9620500eaa5d`.

This is the only change to the shared replay engine. Development behavior and
all recomputed values are unchanged. The Aug 8 support/policy wrappers must
bind the generalized hash before any call.

## Confirmation Components

- producer `scripts/regretbench_deepseek_dynamic_depth2_confirmation.py`:
  `7cbe10ec1dde5406d21dfb2ee02431ca5771e5e760c8bcb939f2cea94ae129d0`;
- independent verifier
  `scripts/regretbench_deepseek_confirmation_result_verify.py`:
  `f8a88ba92a079f2993838ea50cac5fb78546ba66f1428a173927b57c72eaa6aa`.

The producer imports the frozen development policy core and applies a scoped
substitution only for the confirmation cohort, confirmation seed constants,
confirmation maximum request count, and confirmation budget. It disables the
optional Luna baseline. The scope restores every development global after the
run.

The confirmation verifier never imports the confirmation producer or the
development producer. It uses the independent replay engine with separately
hardcoded confirmation cohort and seed constants, reparses all raw supports,
reconstructs hidden truths and official mappings, recomputes selected roots,
aligned terminal metrics, CRN groups, paired bootstraps, mechanics, science,
and status, and additionally verifies the confirmation metadata and absence of
optional-baseline calls.

## Authorization

The producer may run only with a hash-bound literal development `passed`
result whose independent replay status is `verified`. A gated null, mechanics
failure, partial artifact, missing daily reconciliation, hash mismatch, or
failed replay opens zero confirmation calls.

The daily executor itself must be implemented, tested, and hash-bound after
these component hashes exist and before the first confirmation call. It may
only enforce the frozen date, predecessor, catalog, budget, artifact, replay,
and reconciliation rules; it cannot change the scientific contract.

An exact-scale zero-network rehearsal over all 64 confirmation tasks produced
the full 8,256-request planning tree plus realized paths, passed every mechanics
gate, independently replayed every endpoint and bootstrap, and rejected a
tampered reported Brier. This rehearsal made zero model calls and cost `$0`.
