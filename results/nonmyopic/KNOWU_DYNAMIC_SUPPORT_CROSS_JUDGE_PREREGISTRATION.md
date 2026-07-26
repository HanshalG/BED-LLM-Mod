# KnowU Dynamic-Support Cross-Judge Preregistration

Date frozen: 2026-07-26

## Purpose

The GPT-5.4 mechanics gate found one missing-truth support-entry world, but the
same model generated supports, simulated users, and judged truth coverage.
Before any ranking-fidelity or policy experiment, rescore the exact frozen 30
supports with an independent model family.

## Frozen inputs and model

- public mechanics SHA-256:
  `bf433f1ff1b6f9ec2acbf359f384922380dfa21b7a06ff4be65c60ec82e76039`
- private mechanics SHA-256:
  `0f12315cb22ca54cf923c9075b7397dc9cd38a814c7c5762120ec0928ef5cf97`
- scorer: `google/gemma-4-26b-a4b-it`
- temperature 0, thinking/reasoning disabled
- same semantic truth packets, support states, score rubric, and threshold 70
- no support regeneration, user simulation, prompt repair, or reissue

## Gates

First run an exact 10-call synthetic serving gate. It must parse all judgments
with zero reasoning, retry, or forced exit and cost at most $0.05. Failure
forbids confirmation.

Confirmation makes exactly six calls, one per frozen world. It must satisfy:

- six accepted requests and six HTTP attempts;
- zero reasoning tokens, retries, and forced exits;
- all six exact judgment objects parse;
- at least one initially missing truth enters after a question;
- `T1W3` is initially missing and at least one of its originally successful
  operating-system/platform roots recovers truth;
- the maximum cross-judge score gain on those two roots is at least 8 points;
- cross-judge/GPT-5.4 binary presence agreement is at least 0.80 over all 30
  support states;
- confirmation costs at most $0.15.

Passing authorizes a separately preregistered candidate-strategy
ranking-fidelity gate, not a depth sweep or policy claim. Failure leaves the
KnowU first-link result same-model and development-only.
