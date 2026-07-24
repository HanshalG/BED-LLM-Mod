# ClinDiag Deterministic De-Anchored Refresh Gate

Date: 2026-07-24

Status: **preregistered before any serving response.**

## Purpose

The de-anchored temperature-`0.5` prompt changed half of each support under exact
replay. This sole calibration tests whether setting support generation temperature to
`0.0` restores reproducibility without reintroducing an explicit prior-retention
instruction.

## Frozen Protocol

Seed `24298` selected fresh challenging case `23697517` and rare case `rare216`.
Neither appeared in any prior fixed-slot run.

The prompt, support size, path, models, and measurement are unchanged from the failed
de-anchored gate:

1. initial 12-diagnosis support;
2. de-anchored full rebuild after stored `present_illness`;
3. full rebuild after stored `present_illness` plus `lab_1`;
4. exact replay of step 3;
5. one semantic stability audit per case.

The only change is GPT-5.4 support temperature `0.0` instead of `0.5`. GPT-5.4 Mini
measurement remains at `0.0`; reasoning and retries remain disabled.

## Frozen Gate

All prior thresholds remain unchanged:

- exactly 10 physical requests;
- all eight supports size 12;
- exact duplicate prompts;
- zero reasoning and zero retries;
- no full target string in source evidence;
- worse directional semantic overlap at least `0.80` on both cases;
- duplicate truth-score gap at most `0.05` on both cases;
- no parser or runtime failure.

The OpenRouter run ceiling is `$0.50` and projected reservation is `$0.15`.

Passing authorizes one fresh small de-anchored headroom screen. Failure closes the
de-anchored prompt family; no further temperature, overlap, or replay tuning follows.
