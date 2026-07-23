# Animals Belief-Recall Ranker Development Plan

Status: **frozen before any ranker response**. This is development on two
previously observed mechanism traces, not a confirmatory policy result.

## Question

Can a target-blind LLM score which candidate question will induce a more
complete LLM-regenerated belief support after the next answer?

The hidden animal is never included in the scoring prompt. The ranker receives
only the existing question/answer history and, for each candidate, its text,
predictive Yes/No probabilities, and regenerated branch-support sizes. These
are all available before selecting a real query.

## Fixed Development Read

- Source traces: the completed seed-1304 and seed-1305 Animals
  coverage-dynamics probes.
- Ranker: non-thinking `google/gemma-4-26b-a4b-it`, temperature zero.
- One batched response per state, 20 states total, strict bare-JSON score vector.
- Primary descriptive comparison: paired expected hidden-truth coverage of the
  ranker's top candidate versus immediate EIG's top candidate.
- Supporting values: candidate-level Spearman association, active-state
  coverage regret, and win/tie/loss counts.

The source targets were already inspected while developing this prompt, so no
threshold on these 20 states can establish a scientific claim. A useful signal
is a positive paired mean gain, fewer active-state misses, and rank association
that is directionally better than immediate EIG. If those do not hold, stop
this ranker. If they do hold, freeze the exact prompt and thresholds for a
fresh target/seed coverage probe before integrating any policy.

## Integrity And Spend

- Prompt construction uses an explicit allowlist and cannot serialize target or
  truth-coverage fields.
- Raw model-visible payloads and responses are stored for audit.
- OpenRouter run cap: `$0.10`; projected cost: `$0.01`.
- Project ledger before responses: `$40.51526205` of `$110`.

## Format-Only Recovery Amendment

The frozen first invocation made all 20 requests but failed closed before a
ranking endpoint because at least one response was not strict bare JSON. The
failure artifact contains no raw completions because v1 parsed before
serializing them. Usage was 9,938 prompt tokens, 463 completion tokens, zero
reasoning tokens, and `$0.00112158`.

A single v2 recovery is frozen before its responses. It changes only the output
parser to accept one standard Markdown JSON fence around the otherwise
unchanged one-field object, and it persists every raw response plus the failing
index if any parse still fails. Prompt, model, temperature, source hashes,
candidate rows, scoring summary, and `$0.10` cap are unchanged. This is still
development-only and cannot establish a policy claim.
