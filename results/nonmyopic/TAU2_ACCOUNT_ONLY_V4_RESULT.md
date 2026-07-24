# Tau2 Account-Only V4 Development Result

Date: 2026-07-24

## Decision

The account-only development smoke passed, but the twelve-variant formal gate
failed. No trajectory confirmation is authorized.

## Serving Smoke

Run `tau2-account-only-v4-smoke-20260724T212711Z`:

- 2 new prompt variants;
- 12 exact GPT-5.4 requests;
- zero reasoning tokens;
- cost $0.043765;
- every root predicted zero immediate EIG;
- every direct root predicted zero depth-two EIG;
- lookup predicted depth-one EIG 0 and depth-two EIG 1.386294 exactly;
- lookup followed by line details on 2/2 variants; and
- depth two selected lookup on 2/2 while depth one selected it on 0/2.

## Formal Gate

Run `tau2-account-only-v4-formal-20260724T212824Z`:

- 12 previously untouched prompt variants;
- 72 exact requests;
- zero reasoning tokens;
- cost $0.261855;
- score-versus-exact depth-two Spearman: 0.733329;
- depth-two lookup selections: 7/12, threshold 10/12 for V4 and 8/12 in
  the inherited V2 gate;
- lookup using line details on every branch: 8/12, threshold 10/12;
- depth-one lookup selections: 0/12;
- mean top-one regret: 1.386294 nats at depth one versus 0.577623 at
  depth two;
- mean regret improvement: 0.808672 nats; and
- depth two beats depth one on 7/12, threshold 8/12.

The conjunction fails. V4 is a strong partial development result, not a
confirmed policy result.

## Failure Localization

All roots preserve the correct zero immediate information in all twelve
variants. The failure occurs in the branch continuation:

- four variants choose `network_status` after lookup and therefore predict no
  information;
- one variant chooses `line_details` but predicts the same airplane-mode
  outcome for all four account hypotheses;
- six variants recover the exact 1.386294-nat partition;
- one variant recovers a partial 0.693147-nat partition.

The hypothesis support, action descriptions, and physical simulator are
identical across variants. Ticket paraphrase alone changes whether GPT uses
the identifier unlocked by lookup. This identifies branch-policy consistency
as the remaining bottleneck after observational equivalence and entropy
scoring are fixed.

## Scope

V4 was transparently selected after the six-world V3 device-state miss. The
formal failure closes Tau2 account planning; there is no trajectory
confirmation or threshold repair.
