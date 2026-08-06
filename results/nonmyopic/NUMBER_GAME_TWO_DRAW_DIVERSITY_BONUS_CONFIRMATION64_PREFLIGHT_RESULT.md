# Number Game Diversity-Bonus Confirmation-64 Preflight Result

Date checked: 2026-08-06

## Decision

- Block A: `waiting_for_aug7_control`.
- Block B: `waiting_for_block_a`.

These are the correct prospective states. All non-predecessor checks pass, and
neither waiting state authorizes a model call.

## Verified Without Calls Or Writes

- execution amendment SHA-256:
  `4c3e947bf9b68f71668c05593b4a224d609de19be52070aefc0ce989c426b39b`;
- prospective preregistration SHA-256:
  `49b1a8bd783f8cbf54ba561ec55567bc4143af3b4358397cab4840d7d778cc5d`;
- canonical dates/seeds/models/request manifest SHA-256:
  `041994f4de92b573c511414a293c049655a6adec6321189f522246b0f1ea6eba`;
- exact request boundary: `3,680` per block and `7,360` total;
- target block directories, daily ledgers, terminal artifacts, and failure
  artifacts are absent;
- `qwen/qwen3.7-plus` is live with structured output, 1M context, and 131,072
  maximum completion tokens;
- `google/gemini-2.5-flash` is live with structured output, 1,048,576 context,
  and 65,535 maximum completion tokens;
- authenticated balance is `$27.702109737`, above the exact `$5` start gate;
- model calls made: `0`; files written: `0`.

The preflight distinguishes an absent future predecessor from a malformed one.
Once present, the Aug 7 control must independently replay exactly. Before
Block B, the stored Block-A source, mechanics-only authorization, request
count, spend ledger, and verification artifact are independently recomputed in
memory without rewriting the banked artifact.

The authoritative paid command now invokes this same full preflight
automatically on every pristine block before any ledger write or component
call. It requires `ready_without_paid_calls` and reuses the preflight's live
credit snapshot to initialize the ledger. Unit coverage confirms wrong-day,
waiting, and failed preflights write nothing and make no component call, while
completed-block resumptions skip fresh execution and preserve banked results.

## Commands

```bash
set -a; source .env; set +a

/opt/anaconda3/bin/python \
  scripts/number_game_two_draw_diversity_bonus_confirmation64_daily_execute.py \
  --block a --preflight

/opt/anaconda3/bin/python \
  scripts/number_game_two_draw_diversity_bonus_confirmation64_daily_execute.py \
  --block b --preflight
```

Focused validation: `117 passed` across the diversity-bonus and Aug 7 execution
tests.
