# Tau2 Account-Only V4 Development Preregistration

Date frozen: 2026-07-24

## Scope

V4 is explicitly post-V3 development, not a sealed confirmation. It isolates
carrier-account uncertainty by fixing device roaming off and crossing:

- data allowance available/exhausted; and
- account roaming enabled/disabled.

These are four official Tau2 mobile-data worlds. Airplane mode remains on in
every world.

The support is scientifically distinct from the failed six-world V3 gate:
device state is not a latent variable. This is appropriate for an
account-record prerequisite question, but selection occurred after observing
that V3 missed the device-roaming field. A positive V4 gate therefore
authorizes only a separately seeded paired trajectory confirmation.

## Exact Opportunity

Zero-call official simulator values:

- all six root actions have zero immediate EIG;
- every phone-side root has zero legal two-step EIG;
- `customer_lookup` returns one identical record;
- `line_details` has four distinct outcomes; and
- lookup then line details is 1.386294 nats.

## Interface

Full GPT-5.4, reasoning disabled, temperature zero. The policy sees the known
uniform four-hypothesis support and legal actions, but no scores or simulator
outputs. It predicts complete two-step outcome partitions exactly as V2/V3.

## Gates

Serving smoke uses two new prompt variants, exactly 12 requests, cap $0.75:

1. every root has predicted immediate EIG exactly zero;
2. both lookup trees use line details on every branch;
3. depth two selects lookup in both variants;
4. depth one does not select lookup; and
5. all trees are valid, finite, complete, and use zero reasoning.

If smoke passes, twelve previously untouched V2/V3 formal prompt variants use
exactly 72 requests under a $2 cap. The existing rank/regret gates remain, with
complete support defined as 4/4. V4 failure closes Tau2 account planning.
