# UCI Thyroid Blood-Workup Qualification Preregistration

## Source and Motivation

The UCI Thyroid Disease archive (DOI `10.24432/C5D010`) includes 7,200
`ann-thyroid` patients and official acquisition metadata. The released delay file
marks 16 history/demographic variables immediate and TSH, T3, TT4, and T4U delayed.
The group file places those four assays in one group, and the expense file records
their shared blood-collection discount. Frozen source hashes are in
`data/nonmyopic/uci_thyroid/README.md`.

An exploratory seed-24147 screen used 1,000 sampled patients and 50 trajectories.
It found d2-minus-d1 entropy AUC +.1590 nats with 44/0/6 wins/ties/losses; d2
collected blood first in every trajectory while d1 queried medical history. No LLM
response was requested. This registration freezes a fresh full-cohort prior and
fresh truth sample before the formal endpoint.

## Exact Environment

- Empirical uniform prior over all 7,200 released train+test rows.
- Target: three-class thyroid status in the released label column.
- Immediate actions: query age or one of 15 released binary history variables.
- Setup action: collect a blood sample; consumes one round and gives no information.
- After collection: TSH, T3, TT4, and T4U become legal assay queries.
- FTI is excluded because it is absent from the released delay/cost files.
- Age and continuous measurements are discretized into six combined-cohort empirical
  quantile bins; binary features retain their released values.
- Answers, posterior filtering, class probabilities, entropy, and decode are exact
  finite-population operations.

## Frozen Qualification

- 1,000 paired truth rows sampled without replacement; fresh seed 24148.
- 8 executed rounds; exhaustive AUC-aligned d1 and d2 planning.
- Primary endpoint: paired d2-minus-d1 entropy-AUC gain.
- Corroboration: paired d2-minus-d1 truth-class-log-posterior-AUC gain.
- 10,000 paired bootstrap replicates.

The gate requires strictly positive 95% lower bounds for both endpoints, complete
paired legal traces, zero immediate EIG for blood collection, and first-round d2 blood
collection in at least 90% of trajectories. A separate implementation must replay
every trace and independently re-solve every chosen action before the result is banked.
Failure stops this task before any LLM policy work.

## Conditional LLM Stage

A pass authorizes only a separately preregistered non-thinking proposal interface.
Machine code will fix root slots before prompting; the model will emit semantic named
continuations per root, never menu indices or EIG scores. Exact scoring must compare
against matched random, a strong d1 root with exact continuation, and exhaustive d2
before any paired policy endpoint is authorized.
