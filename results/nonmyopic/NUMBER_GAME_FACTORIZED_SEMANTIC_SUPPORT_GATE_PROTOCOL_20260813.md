# Number Game Factorized Semantic-Support Gate Protocol

Date frozen: 2026-08-13, after pushed terminal closure of explicit-mask generation and
before any response under this distinct interface.

## Capability Split

Three prior interfaces isolated two different failures. Large 24-rule responses
truncate, while three eight-rule shards serialize cleanly. However, Qwen-generated
101-bit masks contradict observed answers for 163/240 hypotheses across every shard.

This interface assigns each model one irreducible semantic task:

- Qwen generates ordinary-language hypothesis descriptions only;
- DeepSeek independently translates each description into its exact extension over
  `0..100`, without seeing the history or observed-answer labels;
- Luna independently judges eight probe memberships from each description, without
  seeing DeepSeek's extensions or the history.

Code never invents or repairs support. It validates schemas, deduplicates translated
extensions, filters exact history inconsistency, and compares the independent
translator and auditor. Thus accepted support remains LLM-generated and
description-grounded, while no model is asked to maintain a long semantic rule and
101 indexed bits in the same generation.

## Frozen Cohort And Calls

Fresh synthetic histories, fixed in order:

1. `11=YES`;
2. `11=YES, 33=YES`;
3. `1=NO, 17=YES`;
4. `16=YES, 17=NO`;
5. `19=NO, 38=YES`.

For each history there are two aggregate draws and three eight-description shards per
draw:

- 30 exact `qwen/qwen3.7-plus` nonreasoning proposal requests, seeds
  `202608133000..202608133029`, output cap 1,100 tokens;
- 30 exact `deepseek/deepseek-v4-flash-0731` nonreasoning translation requests, seeds
  `202608133100..202608133129`, output cap 1,200 tokens;
- 10 exact `openai/gpt-5.6-luna` nonreasoning audit requests, seeds
  `202608133200..202608133209`, output cap 1,200 tokens.

There are exactly 70 accepted requests/HTTP attempts, zero retries, and concurrency at
most 30/30/10 by phase. Proposal temperature is `0.7`; translation and audit
temperature are `0.0`.

## Schemas And Privacy

Proposal shards contain exactly eight objects with unique `hypothesis_id`, semantic
`name`, and ordinary-language `description` at most 180 characters. Descriptions may
not contain code, explicit enumeration, exception/lookup language, mask references,
or observed-answer statements. The history is shown only to Qwen so it can propose
plausible consistent rules.

Each DeepSeek translator receives only the eight IDs/names/descriptions and domain
`0..100`, never history or labels. It returns each ID exactly once with a 101-character
binary `membership_mask`. Masks must be nonconstant and unique across the merged draw.
Code rejects, rather than repairs, any translated extension inconsistent with history.

Each Luna audit receives only the merged 24 IDs/names/descriptions and eight
deterministic probe integers per hypothesis. It never receives masks, histories,
labels, or earlier responses. Probes use collision-resolved
`SHA256(interface_version|translator_seed|hypothesis_index|probe_index) mod 101`;
observed integers replace final probes, but their labels are not shown. Luna returns
eight booleans per hypothesis. Semantic validity requires translator/auditor agreement
at least `7/8`.

Private raw output contains all model responses. Public output contains only aggregate
counts, request hashes, extension hashes, and agreement histograms, never descriptions
or masks.

## Gates

All must pass:

- exact 70 clean strict responses, models, seeds, prompt/payload hashes;
- zero retries, provider errors, reasoning tokens, or forced exits;
- every proposal shard exactly eight valid unique descriptions;
- every translation shard covers the same eight IDs exactly once;
- every merged draw exactly 24 unique nonconstant translated extensions;
- every merged draw at least 20 history-consistent extensions;
- every two-draw history pool at least 28 unique history-consistent extensions;
- every second draw contributes at least four new history-consistent extensions;
- pooled translator/auditor agreement at least 90% over 1,920 judgments;
- every draw at least 18 semantically valid and history-consistent extensions;
- every two-draw pool at least 26 semantically valid extensions;
- each history 2--5 at least 12 semantically valid extensions absent from history 1;
- exact observed-answer obedience for every accepted extension.

The protocol intentionally allows some translator outputs to be history-inconsistent,
because the translator is blinded to history. The stated support floors are applied
after this value-free filter and are not lowered after responses.

## Budget And Decision

The stage cap remains `$0.08`; the account-wide Europe/London cap remains `$5.00`.
Authenticated catalog and credits are reread before every phase and HTTP attempt.
Every serialized input byte is conservatively counted as one token, and every
concurrent request reserves maximum exposure. Accepted prior-phase cost plus full next
phase exposure must stay under `$0.08`.

`mechanics_pass` authorizes only a separately frozen fresh scientific protocol with
compute-matched fixed-support, myopic, random, common-random-number, and sealed
endpoint controls. Any failure closes this exact interface, histories, prompts, seeds,
caps, parser, and thresholds. No response may be reused scientifically, and no Number
Game target, source tree, policy endpoint, or prior failed response may be opened.
