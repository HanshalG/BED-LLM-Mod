# ChemBench Costed-Repeat Medium Second Disk-Guard Pause

Date: 2026-08-24 (Europe/London)

## Event

The corrected detached disk guard resumed observation when the host woke and
reported `229,452 KiB` available at `2026-08-24T19:43:57+01:00`. It sent only
`SIGSTOP` to the sole authorized medium process, PID `50494`, and exited. The
process is preserved at state `T+` and CPU `228:59.22` with its exact frozen
command. `medium.json` remains absent.

The runtime files remain present and were not opened as databases:

- `001-proposal-cache.sqlite3`: `747,028,480` bytes, last modified
  `2026-08-21T13:56:23+01:00`.
- `003-transition-audit-cost-blind.sqlite3`: `33,037,541,376` bytes, last
  modified `2026-08-21T13:56:23+01:00`.

Filesystem purgeable-space accounting recovered after the guard fired. Read-only
checks subsequently reported about 9.5-10.4 GiB available, still below the
frozen 12 GiB resume threshold. No `SIGCONT` was sent.

## Cleanup audit

No file was deleted or moved. Read-only measurements found these regenerable
cache candidates:

- `/Users/hanshalgoyal/.cache/uv`: `6,508,064 KiB`.
- `/Users/hanshalgoyal/.cache/codex-runtimes`: `1,624,204 KiB`.
- `/Users/hanshalgoyal/.cache/torch`: `419,924 KiB`.
- `/Users/hanshalgoyal/Library/Caches/pip`: `3,896,532 KiB`.

Approval to delete only the uv cache had already been requested and remains
pending. The process, runtime databases, source checkout, runtime worktree, and
all result artifacts remain untouched. Resume requires explicit cleanup approval,
at least 12 GiB available, and a fresh exact binding/process/file audit.

## Accounting

Authenticated OpenRouter credits/usage/balance remain
`245.000000000 / 220.376693994 / 24.623306006`. No model, HTTP, endpoint, hard
shard, duplicate medium, or paid call was opened. Aug 24 account-wide spend is
`$0.000000`.
