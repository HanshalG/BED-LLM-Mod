# Number Game Bitmask Semantic-Support Gate Protocol

Date frozen: 2026-08-13, after terminal closure of the array-extension interface and
before any response under this distinct interface.

## Motivation And Distinction

The extension-native array gate accepted ten clean Qwen requests with zero retries and
zero reasoning, but three responses reached the frozen 3,300-token ceiling and
contained truncated JSON. The independent terminal audit found seven parseable clean
stops, three length stops, zero audit calls, and no target or endpoint access.

This successor changes the representation and all histories, proposal seeds, and audit
seeds. It does not rerun, repair, or lower any gate on the closed interface. Each LLM
hypothesis still denotes one of `2^101` subsets, but serializes its exact extension as
one fixed 101-character bitmask rather than a variable list of integers.

## Frozen Interface

- proposal model: exact `qwen/qwen3.7-plus`, nonreasoning;
- semantic auditor: exact `openai/gpt-5.6-luna`, nonreasoning;
- exactly 24 hypotheses per proposal response;
- each hypothesis has exactly `name`, `description`, and `membership_mask`;
- `membership_mask[i]` is `1` exactly when integer `i` is in the extension;
- masks match `^[01]{101}$`, are neither all zero nor all one, are unique per draw,
  and obey the supplied observations exactly;
- descriptions remain at most 180 characters and may not contain code, explicit
  enumeration, exception lists, lookup language, or references to the mask.

Fresh histories, fixed in this order:

1. `9=YES`;
2. `9=YES, 18=YES`;
3. `5=NO, 25=YES`;
4. `12=YES, 13=NO`;
5. `6=NO, 30=YES`.

Two proposal draws are made per history. Proposal seeds are
`202608132600..202608132609`; audit seeds are `202608132700..202608132709`.
Concurrency is at most ten per phase, temperature is `0.7`, proposal output is capped
at 3,000 tokens, audit output at 1,200 tokens, and there are exactly twenty accepted
requests/HTTP attempts with zero retries.

## Independent Semantic Audit

The auditor receives only each name, description, and eight deterministic probe
integers. It never receives masks, observation labels, prior responses, Number Game
targets, or endpoints. Probes use
`SHA256(interface_version|proposal_seed|hypothesis_index|probe_index) mod 101` with
deterministic collision resolution; observed integers replace the final probes.

The auditor returns eight booleans per hypothesis. A hypothesis is semantically valid
only at agreement at least `7/8`. Public output contains counts and mask hashes, never
descriptions or masks.

## Gates

All parent support and semantic floors are retained:

- exact transport, models, seeds, payload/prompt hashes, clean stops, strict schemas,
  zero retries/reasoning/forced exits;
- every draw exactly 24 valid unique nonconstant history-consistent masks and hence at
  least 20;
- every two-draw pool at least 28 unique masks;
- every second draw contributes at least four masks;
- zero lexical failures;
- pooled semantic agreement at least 90% over 1,920 judgments;
- every draw at least 18 semantically valid masks;
- every two-draw pool at least 26 semantically valid masks;
- each history 2--5 has at least 12 semantically valid masks absent from history 1's
  pool;
- exact observed-answer obedience.

## Budget And Ordering

The stage cap is `$0.08` and the account-wide Europe/London day cap remains `$5.00`.
Immediately before each phase and every HTTP attempt, authenticated usage and exact
catalog route/prices are reread. Conservative exposure treats every serialized input
byte as one token and reserves the maximum request exposure for every concurrent
request. Proposal exposure must fit the whole stage cap; accepted proposal cost plus
audit exposure must fit the stage cap. Any mismatch fails before the affected phase or
request. No retry or forced-final continuation is permitted.

## Decision

- `mechanics_pass` authorizes only a separately frozen fresh scientific protocol with
  compute-matched fixed-support, myopic, random, common-random-number, and sealed
  endpoint controls;
- any failure closes this exact interface, histories, prompts, seeds, masks, token
  caps, probes, and thresholds.

Mechanics responses cannot be reused scientifically. No Number Game target, source
tree, policy endpoint, or prior failed response may be opened by this gate.
