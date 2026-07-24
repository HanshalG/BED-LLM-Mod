# ClinDiag Multisample Atomic Opportunity Preregistration

Date: 2026-07-24

## Purpose

This gate changes the belief representation rather than repairing a closed
ClinDiag threshold. Earlier single-list refreshes exposed a tradeoff:
prior-retaining supports were reproducible but inert, while de-anchored supports
were responsive but set-unstable. Here a belief state is the empirical
distribution from three independent de-anchored GPT-5.4 differentials.

The gate asks only whether this representation provides a stable, truth-anchored
two-step opportunity. It is an oracle development diagnostic, not a policy,
likelihood model, or holdout result.

## Frozen Data and Actions

The pinned ClinDiag archive, source commit, lexical target-leak filter, and
stored-evidence environment are unchanged. Seed `24325` selects:

- smoke: `26933852`, `rare129`;
- formal: `17700107_1`, `20921516`, `rare269`, `rare295`; and
- unused reserve: `23252529`, `27783909`, `rare257`, `rare153`.

These IDs were selected statically from 418 fresh eligible cases after excluding
every ID named in an earlier ClinDiag script. The 60-case staged-generator
holdout remains sealed.

Every case has the same six generic actions:

1. present illness;
2. family/social context;
3. first physical-examination slot;
4. first laboratory slot;
5. first imaging slot; and
6. first other-test slot.

Selecting an action reveals the corresponding stored archive value. Procedure
names and findings are hidden before selection. No LLM generates a patient
observation.

## Belief Dynamics and Measurement

For the initial state, every one-step state, and every one of 30 ordered
two-action states, full GPT-5.4 nonreasoning independently generates three
12-diagnosis supports at temperature `.5`. The prompt contains the initial
presentation and cumulative evidence in acquisition order, but no prior
differential. This is a full de-anchored refresh.

Only after all supports are generated does GPT-5.4 Mini nonreasoning receive the
hidden diagnosis. It assigns a strict same-diagnosis/synonym match score to
each sample. State value is the mean of the three scores; empirical truth
coverage is the fraction at or above `.80`.

The greedy first action maximizes one-step state value. Its continuation is the
best pair beginning with that action. The oracle pair maximizes value over all
ordered pairs. The non-myopic gap is oracle-pair value minus greedy-continuation
value. One fixed pair per case is independently regenerated in another
three-sample batch as a replay-stability control.

## Serving Smoke

Two fresh cases each generate initial, one-step, two-step, and exact-prompt
replay states, with three samples per state. One joint semantic request scores
the twelve supports for each case. Exact count is 26 requests, projected
`$0.25`, hard cap `$0.75`.

Pass requires both cases, all 24 supports at size 12, exactly 26 requests, zero
reasoning, and finite replay gaps. No scientific threshold is read from smoke.

## Formal Opportunity Gate

Only smoke passage authorizes the four formal cases. Exact count is 608
requests: 456 GPT-5.4 support generations and 152 GPT-5.4 Mini state
measurements. Projected cost is `$3.00`, hard cap `$8.00`.

All gates are conjunctive:

1. all four cases complete, exactly 608 requests, and zero reasoning;
2. mean initial empirical coverage is at most `.50`;
3. at least two cases have one-step coverage spread of at least `1/3`;
4. at least two cases improve best-pair coverage over best-one-step coverage by
   at least `1/3`;
5. at least one case has a non-myopic soft gap of at least `.10`;
6. mean non-myopic soft gap is at least `.03`;
7. mean oracle-pair gain over best one-step is at least `.05`;
8. the oracle first action differs from greedy on at least one case;
9. mean fixed-pair replay soft gap is at most `.10`; and
10. maximum replay soft gap is at most `.20`.

Failure closes this exact multisample fixed-six-action opportunity route. It
does not authorize threshold repair, case replacement, planner construction,
or holdout use. Passage authorizes only a separately preregistered target-blind
ranking-fidelity gate.
