# Number Game Classical-Grammar Irreducibility Audit Result

Status: **inconclusive at the frozen endpoint-power gate; strong
support-generation mechanism evidence**.

Date: 2026-07-29

Model calls and cost: `0` / `$0`.

Independent replay on 2026-08-06 regenerated `RESULT.json` byte-for-byte from
the hash-bound source artifacts (`cmp` exit `0`); the dedicated audit suite
passes `4/4`. No model response or prospective endpoint was opened.

Preregistration:
`NUMBER_GAME_CLASSICAL_GRAMMAR_IRREDUCIBILITY_PREREGISTRATION.md`.

Public result:
`results/nonmyopic/number_game_classical_grammar_irreducibility_audit/number-game-classical-grammar-irreducibility-audit-20260729/RESULT.json`.
SHA-256:
`514945d268712227c128ecb9be9ee0a495cf1682a1c53c99ff5428b8da0bf9a7`.

## Classical Bank

The deterministic grammar produced:

- `451` deduplicated core atoms;
- `3,526` deduplicated extended affine/inverse atoms;
- `608,850` pairwise Boolean candidates;
- `416,366` unique nonconstant extensions after complements; and
- bank SHA-256
  `6e2a523d7d7c0c59b5df0494de37087525c7009d43a5eaba01390e005d85e0e0`.

This passes the frozen minimum of 100,000 extensions by more than fourfold.

## LLM Planning Supports

GPT-5.4 Mini's planning supports remain substantially outside this broad bank:

| Stage | Occurrences outside bank | Unique extensions outside bank |
|---|---:|---:|
| Initial | 33/1,412 (2.34%) | 19/181 (10.50%) |
| Generated first refresh | 2,933/18,217 (16.10%) | 1,565/3,578 (43.74%) |
| Generated second refresh | 5,613/27,822 (20.17%) | 2,981/5,847 (50.98%) |

Every one of the 64 planning trees contains at least one grammar-novel
second-refresh extension: 32/32 in each independent source study. All three
frozen support-novelty gates pass.

This is direct extension-level evidence that path-conditioned LLM regeneration
is not merely selecting from the preregistered classical closure.

## Independent Qwen Endpoints

The independent Qwen endpoint generator is much more classical:

- `208/23,227` endpoint occurrences are outside the bank (`0.8955%`);
- those occurrences represent only `34` unique grammar-novel extensions;
- no tree has the required 8/16 nonempty grammar-novel draws; and
- the observed nonempty-draw range is `0..6`.

All four frozen endpoint-power gates fail. The grammar-novel extensions are
not all one-bit aliases: their nearest-bank Hamming distance has median `2`,
mean `4.91`, maximum `20`, and 22/34 are at least two domain points from the
bank. There are nevertheless too few independently generated target
occurrences to support the preregistered policy test.

## Efficacy Status

The fixed depth-three versus depth-two efficacy comparison on grammar-novel
targets remains unopened. The preregistration requires at least 48 analyzable
trees with 8 nonempty draws each; the observed count is zero. We do not pool
sparse draws, lower the threshold, narrow the grammar, or report a post-hoc
underpowered score.

The audit is therefore **inconclusive**, not negative:

- the LLM planning belief dynamics clearly generate support outside a large
  classical bank;
- the existing cross-judge endpoint protocol does not sample enough concepts
  outside that bank to establish that the non-myopic gain is concentrated
  there; and
- the already reported all-target depth-three result is unchanged, but it
  should not be described as a direct powered defeat of this enumerated bank.

Per the frozen decision rule, this route authorizes no paid follow-up or grammar
repair. The next paid experiment should use a task whose independent endpoint
distribution is intrinsically semantic rather than prompt-biased toward
compact arithmetic predicates.
