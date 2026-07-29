# LongVid Robust Keyed First-Link Mechanics Preregistration

Date frozen: 2026-07-29

## Purpose

The LongVid four-hop opportunity replicated on an untouched 40-video block:
10/40 tasks require an immediately weaker first search to recover one more
necessary clip after four searches. Previous LLM-native mechanics never reached
a complete scientific score. They failed on JSON/line syntax or one truncated
HTTP response.

This protocol prospectively fixes transport only:

- order-insensitive keyed pipe rows replace order-sensitive lines;
- exact keys, row counts, field counts, numeric domains, uniqueness, grounding,
  and semantic constraints remain fail-closed;
- at most two adapter-level provider retries are allowed;
- no semantic correction, parser extraction, completion repair, or model
  reissue is allowed.

This is a new mechanics gate, not a reinterpretation or rerun of a failed
artifact.

## Model And Role

- model: `openai/gpt-5.4`
- BED reasoning: disabled
- temperature: `0`
- maximum output: 900 tokens
- concurrency: 8

The model generates six weighted open semantic evidence-chain hypotheses,
regenerates them after each retrieved caption, chooses the highest-weight next
semantic search, and ranks blinded one-step versus four-step belief
trajectories. BM25 only executes the generated searches. Hidden evidence clips
are unavailable until both choices are frozen.

## Exact 10-Call Smoke

The public synthetic fixture from the prior LongVid transport smoke is used
with fresh responses:

- 1 initial support;
- 8 observation-conditioned refreshes;
- 1 final trajectory rank.

All 10 accepted outputs must parse; all eight refreshes must change and be
unique; each refresh must have at least four valid observation-grounded
anchors; the rank must parse; reasoning/forced exits must be zero; accepted
requests must equal 10; HTTP attempts may be 10--12 with at most two logged
transport retries; cost must not exceed `$0.20`.

Only a full smoke pass authorizes mechanics.

## Mechanics Pairs

The structural confirmation artifact is hash-bound at
`895e448c047ca7afa924393da0a4f637a21735c8a770336fa17db0a72fdd4081`.
After excluding all 18 LongVid rows previously sent to a model, exactly two
unused confirmed strict pairs remain:

| Row | Immediate root | Four-hop oracle root | Immediate/final clips |
|---:|---:|---:|---:|
| 319 | 0 | 9 | 2 / 3 |
| 120 | 2 | 1 | 2 / 3 |

The layout hash is
`a2783ca4a1a1140e427e031820eddc0a2e055582ab4c4fb03fc68dd149c46a9f`.
These endpoints were already opened by the zero-call confirmation, but neither
row has received a model request.

Mechanics makes exactly 22 accepted requests: two initial supports, 16
path-conditioned refreshes, two blinded immediate ranks, and two blinded final
ranks. Hidden evidence IDs are loaded only after all ranks are checkpointed.

## Frozen Mechanics Gate

Every check must pass:

- 22 accepted requests, 22--24 HTTP attempts, and at most two provider retries;
- zero reasoning tokens and forced exits;
- every output parses, all four paths contain four distinct retrieved clips,
  at least 14/16 refreshes change, and all 16 have at least four valid anchors;
- both pairs are endpoint-rankable;
- final ranks choose the higher-coverage root on 2/2;
- final pairwise accuracy strictly exceeds immediate accuracy;
- both final choices are correct changes from the immediate choices;
- total final-minus-immediate necessary-clip coverage is exactly `+2`;
- cost is at most `$0.50`.

Random choice remains descriptive at this two-pair mechanics scale.

## Decision Rule

A full pass authorizes design and preregistration of a target-blind policy on
all 22 untouched reserve videos. It does not itself establish policy efficacy.
Any smoke or mechanics failure closes this robust LongVid interface for this
cycle; there is no model, prompt, row, parser, or threshold repair.

OpenRouter displayed `$12.757678083` before freezing. The user has authorized
aggressive four-day spending with no reserve, but the reported latest top-up
was not yet visible. OatML remains unused.
