# OpenRouter Five-Dollar Daily Plan

Date frozen: 2026-08-06

## Account And Rule

Authenticated state at freeze:

- total credits: `$245.000000`;
- total usage: `$217.297890`;
- available balance: `$27.702110`.

The newly reported `$30` top-up is not yet visible on this API key, so it is
not budgeted until the credits endpoint posts it.

The hard rule is `$5.00` per Europe/London calendar day, account-wide, with no
rollover. Every paid runner checks cumulative live usage against a frozen
opening baseline before its first provider call. Concurrency changes throughput,
not the spend ceiling.

## Four-Day Allocation

| London date | Primary purchase | Expected / maximum | Decision rule |
|---|---|---:|---|
| Aug 6 | Fresh Qwen source-32 | `$4.2604` actual / `$5.00` | Complete; no more paid calls today |
| Aug 7 | Sealed Qwen history-blind control | about `$3.21` / `$4.25` | Run first and alone; preserve the already frozen control ledger |
| Aug 8 | Prospective two-draw diversity-bonus confirmation-32 | about `$4.26` / `$5.00` | Exact fresh seeds and coefficient; refuse if any earlier account spend reduces the full allowance |
| Aug 9 | Luna + DeepSeek-0731 reliability-128 gate, then gated model-native work | at most `$0.20` for gate / `$5.00` total | Continue only with models that pass strict conditioned-support floors |

The first four days spend at most `$20`. The currently authenticated balance
therefore leaves at least `$7.70` for Aug 10 onward even if every daily cap is
fully used. A posted top-up extends the schedule but does not raise any daily
cap.

## Model Roles

### Qwen 3.7 Plus

Qwen remains the paper-critical generator. It has passed exact structured
serving and produced the existing 96-tree and fresh 32-tree results. It buys
the Aug 7 control and Aug 8 prospective monotonic-depth test.

### GPT-5.6 Luna

Luna is a budget candidate, not yet a replacement. Its current price is
attractive and its exact-10 support generation passed, but the scale attempt
had `15/1,200` forced-length responses and a fatal JSON failure. The 128-item
gate tests whether that was a manageable tail or a real reliability limit.

### DeepSeek V4 Flash 0731

DeepSeek-0731 is the cheapest serious candidate, but its exact-10 conditioned
test included a `0/24` valid response. It must pass the same strict
history-conditioned support gate before receiving any policy-scale budget.
Low price cannot compensate for belief-support collapse.

## Concurrency

- use aggregate concurrency `64` for the two 128-item budget-model gates;
- retain the validated Qwen concurrency for sealed paper runs;
- raise toward `128` or `256` only after a mechanics-clean block shows that
  provider throttling and forced exits are not increasing.

The gate workload is too small for concurrency `256` to improve useful
throughput, and changing serving pressure during a sealed replication adds no
scientific value.

## Aug 9 Branch

If one budget model passes, spend the remainder of Aug 9 on a small paired
LLM-native planning block using that model, with strict accounting and the
same semantic-support endpoints. If both pass, choose by conditioned-support
quality first and expected cost second. If both fail, close them for paper
work and use the remainder only on a Qwen experiment that fits the remaining
allowance; do not relax parsers or support floors to manufacture a cheap pass.

## Aug 10 Onward

- If Aug 8 passes, buy an independent Qwen replication of the frozen selector.
- If Aug 8 is null, do not repeat it unchanged; use the best mechanics-passing
  budget model to investigate a new LLM-native belief-dynamics intervention.
- Keep one experiment decision per day and bank its result before opening the
  next paid block.
