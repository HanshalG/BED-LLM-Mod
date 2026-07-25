# HotpotQA Directional Unlock Audit Result

Date: 2026-07-25

## Outcome

**Every frozen zero-call gate passed.** The environment has a large,
externally annotated directional-order opportunity and authorizes a separate
LLM-native belief-dynamics smoke.

Public artifact:

`results/nonmyopic/hotpot_directional_unlock_opportunity/AUDIT.json`

## Results

| Metric | Result |
|---|---:|
| Opportunity bridge questions | 500 |
| Strict directional unlocks | 166 |
| Strict unlock rate | 33.2% |
| Medium or hard strict unlocks | 166 |
| Neither support title verbatim in question | 55 |
| Title-only BM25 misses enabling root | 47 |
| BM25 selects enabling / answer / distractor | 119 / 14 / 33 |
| Enabling-first support coverage | 2 |
| Answer-first support coverage | 1 |
| OpenRouter calls and cost | 0 / $0 |

The main exclusions were:

| Exclusion | Count |
|---|---:|
| Enabling paragraph lacks a sole answer-title link | 217 |
| Answer phrase not unique to one support's annotated sentences | 89 |
| Reverse support link exists | 21 |
| Yes/no or otherwise excluded answer | 7 |

## Interpretation

The passed cohort is stronger than a generic multihop dataset claim. Every
included task has the same externally checked intervention:

1. inspect the enabling support paragraph;
2. its only mention of another candidate title identifies the answer support;
3. inspect that title and recover both annotated support documents.

Starting with the answer-bearing paragraph recovers immediate answer evidence
but does not expose the enabling support, so the same deterministic link
transition covers only one support document. This is a one-document
non-myopic gap under a target-independent transition rule.

The opportunity is also not confined to questions that print their support
titles: 55 strict cases contain neither title verbatim, and a title-only BM25
root misses the enabling document on 47 cases.

## What This Does Not Show

This audit uses gold support annotations to define and evaluate the cohort. It
does not show that an LLM can identify the enabling root, refresh useful
hypotheses after reading it, or beat a myopic semantic baseline.

The next smoke must isolate that causal chain. Branch hypothesis generation and
continuation scoring will be separate calls. The scorer will receive blinded
belief states and candidate titles but not the question or revealed paragraph,
preventing it from bypassing the generated belief state. Development and
holdout endpoints remain sealed.
