# Number Game Extension-Native Semantic-Support Gate Protocol

Date frozen: 2026-08-13, before any response under this interface.

## Motivation

The fully fresh Qwen history-blind control failed because one of 3,072 strict draws
contained 11 invalid Python expressions. A prospectively frozen label-free audit found
that this was the only draw below 16 valid hypotheses and the only pool below 24; 99%
of draws had at least 20 valid hypotheses and 99% of pools at least 25.

This gate tests a genuinely new belief representation rather than lowering a floor or
rerunning the failed cohort. The LLM emits semantic rule descriptions together with
their exact extension over integers 0 through 100. Code validates the extension, and
an independent model reconstructs memberships from the description alone. Only
description-extension pairs that pass semantic agreement may enter support.

## Scientific Role

The extension space has `2^101` possible subsets and is not enumerated by the method.
The proposal model still supplies the open hypothesis support; the auditor tests that
the extension is entailed by the semantic description rather than an arbitrary lookup
table. Code owns only validation, deduplication, consistency filtering, likelihoods,
and BED scoring.

This is a value-free mechanics and semantic-calibration gate. It uses synthetic
histories fixed below and opens no Number Game target, source tree, policy endpoint,
or prior failed response.

## Models And Requests

- proposal model: exact `qwen/qwen3.7-plus`, nonreasoning;
- semantic auditor: exact `openai/gpt-5.6-luna`, nonreasoning;
- five synthetic histories, fixed in this order:
  1. no observations;
  2. `8=YES`;
  3. `8=YES, 16=YES`;
  4. `7=NO, 21=YES`;
  5. `10=YES, 11=NO`;
- two independent proposal draws per history: 10 proposal requests;
- one audit request per proposal draw, checking all 24 descriptions at eight
  deterministic hash-derived membership probes: 10 audit requests;
- exactly 20 accepted requests and exactly 20 HTTP attempts; zero retries;
- proposal seeds `202608132400..202608132409`;
- audit seeds `202608132500..202608132509`;
- concurrency at most 10 per phase;
- proposal output cap 7,000 tokens; audit output cap 3,000 tokens;
- stage cost cap `$0.08`; full worst-case exposure reserved before dispatch.

Catalog prices and the authenticated account-wide Europe/London ledger must be reread
immediately before dispatch. Any route, price, balance, usage, or allowance mismatch
fails before calls.

## Proposal Schema

Each proposal response contains exactly 24 objects with exactly:

- `name`: concise unique semantic rule name;
- `description`: one coherent general rule, at most 180 characters, without explicit
  enumeration, exception lists, code, or references to the requested membership bit
  string;
- `members`: strictly increasing unique integer array, values in `0..100`.

The parser rejects empty/all-domain extensions, duplicate extensions, descriptions
containing code-like or enumeration language, and any extension inconsistent with the
history. There is no syntax compilation and no semantic repair.

## Independent Semantic Audit

For each proposal draw, the auditor receives only the 24 names/descriptions and eight
probe integers per hypothesis. It returns one boolean membership judgment for every
hypothesis-probe pair. It does not receive the proposed member arrays, history labels,
source targets, or endpoints.

Probe integers are generated before responses from
`SHA256(interface_version || proposal_seed || hypothesis_index || probe_index)` modulo
101, with deterministic collision resolution. For histories with observations, the
observed integers are included in the eight probes and replace the final hash probes.

A hypothesis is semantically valid only if auditor agreement with the proposed
extension is at least `7/8`. The public result stores aggregate agreement counts and
extension hashes, never descriptions or member arrays.

## Gates

All must pass:

### Serving

1. exact 20 accepted/HTTP requests, zero retries/provider errors;
2. all responses finish cleanly with nonempty strict schemas;
3. zero reasoning tokens and forced exits;
4. exact model IDs, seeds, prompt hashes, and payload hashes;
5. cost at most `$0.08` and account-wide daily spend at most `$5.00`.

### Proposal Mechanics

1. all 10 draws contain exactly 24 schema-valid items;
2. every draw has at least 20 unique nonconstant history-consistent extensions before
   semantic auditing;
3. every two-draw pool has at least 28 unique extensions before semantic auditing;
4. every second draw contributes at least four new extensions;
5. no description fails the no-code/no-enumeration lexical gate.

### Semantic Calibration

1. at least 90% pooled auditor agreement over all 1,920 judgments;
2. every draw has at least 18 semantically valid unique extensions;
3. every two-draw pool has at least 26 semantically valid unique extensions;
4. every history has at least 12 semantically valid extensions not present in the
   no-observation pool;
5. observed-answer obedience is exact for every accepted extension on histories 2--5.

## Decision

- `mechanics_pass`: authorize only a separately frozen fresh scientific protocol using
  this exact representation, models, prompts, parser, and auditor. That protocol must
  retain compute-matched fixed-support, myopic, random, CRN, and sealed endpoint gates.
- otherwise: close this interface and do not repair prompts, seeds, token caps, probes,
  thresholds, or parsers on these histories.

No response from this mechanics gate may be reused in a scientific cohort. The failed
August 7 control and every prior result remain immutable.
