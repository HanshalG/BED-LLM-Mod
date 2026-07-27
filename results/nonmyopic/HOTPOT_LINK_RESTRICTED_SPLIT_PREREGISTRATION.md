# HotpotQA Link-Restricted Split Preregistration

## Purpose

Freeze fresh data boundaries for a corrected Hotpot directional-unlock
environment before opening any new endpoints.

V4 revealed that the runner offered every remaining context title after every
root, while the source qualification defines the transition through titles
actually mentioned in the revealed paragraph. The corrected environment will
offer only paragraph-mentioned context titles, plus an explicit stop action
when none are mentioned.

The five opened V4 development rows may be used only for mechanics design.
They cannot support a corrected efficacy claim.

## Source And New Splits

Start from the still-sealed 71,391-row holdout of the frozen official
HotpotQA converted train shards. Preserve its existing deterministic order and
slice without reading endpoint columns:

| Split | Rows | Purpose |
|---|---:|---|
| mechanics | 100 | transport and action-graph design |
| development | 500 | select first 20 strict tasks for model development |
| confirmation | 2,000 | powered fresh confirmation only after development |
| retained holdout | 68,791 | untouched |

The manifest stage reads only `id`, `type`, and `level`, verifies the old
holdout ordered-ID hash, and emits only counts and hashes.

## Zero-Call Opportunity Screen

Only after the manifest passes, materialize endpoint columns for mechanics
100 and development 500. Apply the already-frozen `strict_unlock` and top-four
qualification exactly. For each qualifying row, construct root follow-up
actions from context titles literally mentioned in that root paragraph.

Require:

- at least three qualifying mechanics rows;
- at least 20 qualifying development rows;
- every qualifying task has exactly one optimal root under exact two-document
  support coverage;
- that unique optimum is the enabling root with value `2`;
- confirmation and retained-holdout endpoint rows remain unopened; and
- zero model calls.

Select the first 20 qualifying development rows in frozen split order. No
prompt, policy, or model threshold is authorized until this structural screen
passes and a separate mechanics protocol is frozen.

## Budget

Manifest and opportunity cost `$0`. Authenticated balance before the screen is
`$29.287059594`; there is no fixed reserve. OpenRouter only; no OatML or
Slurm.
