# ChemBench Costed-Repeat Medium Disk Guard Pause

Date: 2026-08-19 (Europe/London)

## Status

The sole authorized disk-backed medium shard was reversibly suspended with
`SIGSTOP` before filesystem exhaustion. This is an operational pause only: no
scientific setting, response, policy, seed, cache record, or result was changed.

## Frozen Process

- PID: `50494`
- process state after guard: `T+`
- accumulated CPU time: `116:03.55`
- runtime implementation commit: `f3eb692346912f8dfa6e9c68e41fe94e638b59a8`
- scientific implementation commit: `d9559a2f03d415966200d9f74c3bd84bbe12f021`
- completed phases: primary `d3`, `d2`, and `d1`
- active phase at suspension: cost-blind `d3`
- terminal `medium.json`: absent
- model calls: `0`
- network calls: `0`
- cost: `$0`

## Guard Evidence

The detached guard checked free space once per minute and was prospectively set
to send only `SIGSTOP` when available space was at or below 6.5 GiB.

```text
2026-08-19T23:54:33+0100 available_kib=6931444
2026-08-19T23:55:33+0100 available_kib=6524512
2026-08-19T23:55:33+0100 disk_guard_triggered available_kib=6524512
50494 T+    29.7 431232 09:59:15 116:03.55
```

The process and its SQLite files must remain untouched until adequate space is
freed. Resume only with `SIGCONT` after rechecking the source/runtime bindings,
process identity, database files, and at least 12 GiB free space. Do not launch
a duplicate medium or the hard shard.

## Regenerable Space Candidates

Read-only inspection found these local cache directories (reported `du` sizes
use 512-byte blocks):

- `~/.cache/huggingface`: `67,306,592` blocks, about 32.1 GiB
- `~/.cache/uv`: `13,016,128` blocks, about 6.2 GiB
- `~/.cache/codex-runtimes`: `3,248,480` blocks, about 1.5 GiB
- `~/.cache/torch`: `839,848` blocks, about 410 MiB

No cache was deleted. Destructive cleanup requires the user's explicit approval.
