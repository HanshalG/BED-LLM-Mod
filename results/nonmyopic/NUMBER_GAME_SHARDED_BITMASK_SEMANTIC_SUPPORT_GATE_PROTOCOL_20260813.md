# Number Game Sharded-Bitmask Semantic-Support Gate Protocol

Date frozen: 2026-08-13, after independent terminal closure of the unsharded bitmask
interface and before any response under this distinct interface.

## Rationale

The unsharded bitmask interface reduced extension serialization but all ten
24-hypothesis Qwen responses still reached 3,000 tokens and truncated. This establishes
that response cardinality, not integer-array encoding alone, is the serving bottleneck.

This successor preserves 24 hypotheses per draw but obtains them through three
independent eight-hypothesis shards. It uses fresh histories and seeds and cannot read
either closed interface's responses. No support, semantic, or authority threshold is
lowered.

## Frozen Requests

- proposal: exact `qwen/qwen3.7-plus`, nonreasoning;
- audit: exact `openai/gpt-5.6-luna`, nonreasoning;
- five fresh histories in order:
  1. `4=YES`;
  2. `4=YES, 20=YES`;
  3. `3=NO, 27=YES`;
  4. `14=YES, 15=NO`;
  5. `2=NO, 22=YES`;
- two aggregate draws per history;
- three independent shards per draw, exactly eight hypotheses per shard;
- proposal seeds `202608132800..202608132829`, ordered by aggregate draw then shard;
- one full-draw semantic audit after merging each three-shard draw;
- audit seeds `202608132900..202608132909`;
- exactly 30 proposal plus 10 audit accepted requests/HTTP attempts, zero retries;
- concurrency at most 30 proposals and ten audits;
- temperature `0.7`, proposal cap 1,300 tokens, audit cap 1,200 tokens.

Each proposal item contains exactly a unique semantic `name`, ordinary-language
`description` at most 180 characters, and exact 101-character `membership_mask`.
Bit `i` denotes membership of integer `i`. Masks are binary, nonconstant, unique
within the entire merged draw, and exactly history-consistent. Descriptions may not
contain code, enumeration, exception/lookup language, or mask references. Shard index
is supplied only as a diversity nonce; it defines no rule family or finite support.

## Semantic Audit And Gates

The auditor sees the merged 24 names/descriptions and eight deterministic probes per
hypothesis, never masks or observation labels. Probes use the same collision-resolved
SHA256 construction under this new interface version and proposal seed tuple; observed
integers replace final probes. Semantic validity requires at least `7/8` agreement.

All gates are unchanged at aggregate-draw level:

- exact 40 clean strict responses, models, seeds, prompt/payload hashes;
- zero retries, reasoning tokens, forced exits, or provider errors;
- each shard exactly eight valid unique items;
- each merged draw exactly 24 valid unique masks;
- each two-draw history pool at least 28 unique masks;
- every second draw contributes at least four masks;
- pooled audit agreement at least 90% over 1,920 judgments;
- each draw at least 18 semantically valid masks;
- each two-draw pool at least 26 semantically valid masks;
- each history 2--5 at least 12 semantically valid masks absent from history 1;
- exact observed-answer obedience.

## Budget And Decision

The stage cap remains `$0.08` and the account-wide London-day cap `$5.00`. Exact live
catalog and credits are reread before every phase and request. Every serialized input
byte is conservatively counted as one input token; maximum per-request exposure is
reserved for every concurrent request. Accepted proposal cost plus full audit exposure
must fit `$0.08`. Any race or mismatch fails closed.

`mechanics_pass` authorizes only a separately frozen fresh scientific protocol with
compute-matched fixed-support, myopic, random, CRN, and sealed endpoints. Any failure
closes this exact interface without retries, prompt/cap changes, or reuse. No Number
Game target, source tree, policy endpoint, or prior response may be opened.
