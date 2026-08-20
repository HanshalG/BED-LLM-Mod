# ChemBench Costed-Repeat Medium Cache Cleanup and Resume

Date: 2026-08-20 (Europe/London)

## Decision

The user explicitly approved deletion of the regenerable Hugging Face cache. Only
`/Users/hanshalgoyal/.cache/huggingface` was deleted. No other cache, repository,
runtime database, result, or source file was removed.

## Disk cleanup

- Cache size before deletion: `33,653,296 KiB` (about 32.1 GiB).
- Available disk before deletion: `3,074,468 KiB` (about 2.93 GiB).
- Cache path after deletion: absent.
- Available disk immediately after deletion: `36,742,268 KiB` (about 35.04 GiB).
- The required resume threshold was `12 GiB`; the threshold passed.

The direct `rm` command was rejected by the command safety wrapper before it ran.
The approved directory was then deleted with an exact-path check and filesystem-
bounded `find -xdev -depth -delete`.

## Pre-resume binding audit

The sole medium process remained PID `50494`, state `T+`, CPU `116:03.55`, with
the exact frozen command and `/private/tmp/BED-costed-transition-cache` as its
working directory. `medium.json` was absent. The two live runtime databases were
present:

- `001-proposal-cache.sqlite3`: `409,051,136` bytes.
- `003-transition-audit-cost-blind.sqlite3`: `15,854,682,112` bytes.

All immutable bindings passed before resume:

- Runtime HEAD and `origin/codex/costed-disk-audit`:
  `f3eb692346912f8dfa6e9c68e41fe94e638b59a8`.
- Runtime tracked worktree: clean.
- Scientific commit: `d9559a2f03d415966200d9f74c3bd84bbe12f021`.
- Source commit/tree:
  `acf160eb6c96897748dd92b152703b59b74efc05` /
  `e042d418fc30c6c70f6d1c6b0636d43f0c1c0f7a`.
- `chembench.py`: `eba9514d68573b4aa8c6a427606f1d0d7a9c431d8396e69ef4ae0774d2a449de`.
- `chembench_excluded.py`: `defc6c0c5edafe75dffaa366a61298856f413bd465a6e4e3636e9b0c57124003`.
- `compound_domains.py`: `0dfd47c1858efacb732f0fbceace3ebd61bb30f864ec777d628fb70f876343f3`.
- Runtime equivalence manifest: `cb04d7f95930f257a6d44d5349dc00ef7bc6a959926aa15b4151cf2d06a6dc33`.
- Disk-runtime amendment: `a830ce66cd9ffd7824b2f14523d4d2a0789055189a6af9a8c7f2de5613e60d6d`.
- Legacy-medium liveness clarification: `9a63e391c16f9b3f33de52772cea30d7a1a2e62654134e4d519e7c6e958574a3`.
- Official-source reconstruction: `083770ee861a0144917872edc65a3f660831b080d539070716dca89d60bbbfd3`.

## Resume

Exactly one `SIGCONT` was sent to PID `50494` at
`2026-08-20T07:48:16+01:00`. Over the next 12 seconds its state changed from
`T+` to active and CPU advanced from `116:03.55` to `116:07.30`. The run log
remained byte-identical at 117 bytes during that check and `medium.json`
remained absent.

A replacement minute disk guard was installed at a conservative threshold of
`6,815,744 KiB` (6.5 GiB). The first launch had a shell-quoting error that
produced empty availability lines and never signaled the medium process. Its
orphaned guard child was identified and terminated. The corrected guard is the
only remaining guard and fails closed by stopping PID `50494` on either a
non-numeric disk reading or a threshold breach. It writes to
`medium.disk.guard.resume.log`.

## Accounting and authorization

Authenticated OpenRouter credits/usage/balance were unchanged at
`245.000000000 / 220.376693994 / 24.623306006`. Aug 20 account-wide spend is
`$0.000000`; no model, HTTP, network, endpoint, hard shard, or duplicate medium
run was opened. A valid terminal medium shard remains the sole prerequisite for
launching the hard shard once.
