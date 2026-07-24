# ClinDiag Fixed-Slot One-Step-Omission Prevalence Screen

Date: 2026-07-24

Status: **preregistered before any model response.**

## Purpose

The four-case pilot found no two-step headroom, but it is too small to determine
whether all-one-step truth omissions occur naturally in the 452-case fixed-slot pool.
This screen applies the unchanged initial-plus-eight-one-step protocol to 20 fresh
cases. It does not generate ordered pairs or evaluate a policy.

## Frozen Cases

Seed `24296` selected 10 challenging and 10 rare cases, disjoint from all six
fixed-slot cases previously used:

`14769677`, `31597024`, `15249641`, `22512486`, `16394081`, `24283228`,
`4003602`, `22913686`, `20560906`, `20685884`, `rare185`, `rare287`,
`rare267`, `rare79`, `rare66`, `rare160`, `rare207`, `rare74`, `rare243`,
and `rare59`.

Selection was mechanical and fixed before model calls.

## Protocol

For every case:

1. GPT-5.4 non-reasoning generates one 12-diagnosis initial support;
2. it independently refreshes that support after each of the same eight generic
   stored-evidence actions;
3. GPT-5.4 Mini non-reasoning scores truth equivalence for all nine supports in one
   post-generation call.

Truth, final diagnosis, title, and answer options remain absent from every generator
payload. The generator uses temperature `0.5`; the judge uses `0.0`; retries are zero.
Raw audit responses are always persisted.

Expected load and budget:

- exactly 200 physical requests;
- OpenRouter run ceiling `$1.50`;
- projected ledger reservation `$0.75`;
- live provider credits and the stricter project ledger checked immediately before
  launch.

## Frozen Gate

A case has `two_step_room` only if its initial truth score and maximum one-step truth
score are both below `0.80`.

All gates must pass:

1. exactly 200 requests;
2. zero reasoning tokens and zero retries;
3. all 180 supports parse to 12 diagnoses;
4. no full hidden-target string occurs in source evidence;
5. at least four of 20 cases have `two_step_room`;
6. no parser or runtime failure.

At least four cases corresponds to a minimum observed prevalence of 20%, enough to
justify an exhaustive ordered-pair screen on a fixed subset. Fewer than four closes
the coarse eight-slot route; no threshold tuning, pair generation, likelihood model,
or policy follows.
