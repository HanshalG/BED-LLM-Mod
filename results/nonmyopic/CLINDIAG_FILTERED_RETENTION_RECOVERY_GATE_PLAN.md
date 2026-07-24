# ClinDiag Filtered-Retention Recovery Gate

Date: 2026-07-24

Status: **preregistered before any serving call.**

## Purpose

The zero-retry filtered-retention gate produced a coherent informative transition and
recovered an omitted truth, but it could refill only 8 and 6 of 12 hypotheses. This
follow-up tests the recovery mechanism specified by BED-LLM rather than changing the
filter threshold: repeat independent generate-filter cycles when constructing the new
pool.

BED-LLM uses threshold `0.20`, retains consistent old beliefs, and repeats generation
up to three rounds when the pool is undersized. This gate fixes three candidate batches
for every path so request count and compute are identical even when old beliefs survive.

## Frozen Cases And Evidence

Seed `24300` selected two fresh fixed-slot cases after excluding every earlier
fixed-slot case:

- challenging `27223150`;
- rare `rare130`.

Both receive the same generic `lab_2` action and its stored archive observation. The
hidden target never enters generation or filtering.

## Update

For each case:

1. generate 12 initial diagnoses;
2. retain old diagnoses with `p(lab_2 | diagnosis) >= 0.20`;
3. independently generate three 12-diagnosis candidate batches from initial
   presentation plus `lab_2`, without showing the old support;
4. retain a candidate only if every visible evidence-item likelihood is at least
   `0.20`;
5. merge retained old hypotheses followed by all valid new batches, lexical-dedupe,
   and cap at 12;
6. repeat steps 3-5 through an independent adapter using byte-identical candidate
   prompts;
7. measure truth equivalence for initial, final, and replay supports.

No adaptive structured retry or post-failure threshold change is allowed. The three
batches are part of the frozen scientific method.

## Models, Calls, And Cost

- generation: `openai/gpt-5.4`, reasoning disabled, candidate temperature `0.5`;
- filtering/audit: `openai/gpt-5.4-mini`, reasoning disabled, temperature `0`;
- exact requests: `30`;
- run ceiling: `$0.50`;
- projected reservation: `$0.15`;
- live balance checked immediately before launch; use the stricter live/ledger
  remainder.

## Frozen Pass Rule

All conditions must hold:

1. exactly 30 requests, zero reasoning, no parser/runtime failure;
2. byte-identical original/replay candidate prompts;
3. both final and replay supports have exactly 12 diagnoses;
4. at least one case has a substantive transition in both paths: at least 4 old
   diagnoses pruned and at least 4 new diagnoses introduced;
5. original/replay truth-score gap is at most `.05` for both cases;
6. neither path loses truth coverage by more than `.05` relative to its initial
   support;
7. no full hidden target occurs in visible source evidence.

Exact normalized set overlap is reported descriptively only. The preceding aggregate
LLM overlap judge was invalid, and low sample identity need not imply a different
underlying belief distribution. Passing authorizes a separate semantic-likelihood and
ranking-fidelity gate, not a policy run.
