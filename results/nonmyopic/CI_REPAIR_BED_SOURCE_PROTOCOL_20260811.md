# CI-Repair-Bench LLM-Native BED Source Protocol

Frozen: 2026-08-11 Europe/London, before reading any selected log, repair diff,
changed-file value, error-type value, or model response.

## Claim Boundary

This is a source-opportunity gate for a new sequential diagnosis environment.
It does not tune or reopen any closed RegretBench, Bongard, LogDx, or other
interface. Passing authorizes only a separately implemented semantic mechanics
gate. It does not authorize a paid policy experiment or endpoint claim.

Each episode is one repository and has a uniform latent state over independently
hash-selected real CI failures from that repository. A diagnostic action returns
a deterministic view of the latent failure's workflow or logs. The source
objective is reduction in entropy over the latent CI failure. A later semantic
interface may present anonymized candidate repair hypotheses, but no repair
outcome is visible to the source opportunity audit.

## Immutable Source

- repository: `CI-REPAIR-BENCH`;
- repository commit: `938310f45d5b76dfed56e2cfd7be344aed5b3de1`;
- Hugging Face dataset commit:
  `7f5a6e8799ff57b9590cc8e87ce220c880d82c7b`;
- Parquet SHA-256:
  `11caa322466b50f0ec31348d6adca096c7f5aac2bc8b9ec8df7721cf5b53d3f9`;
- expected rows / distinct `repo_name` values: `567 / 103`.

The split builder may project only `id` and `repo_name`. It must not load
`workflow`, `logs`, `diff`, `changed_files`, `error_type`, commit links, or any
other content or outcome column.

## Repository-Disjoint Split

A repository is eligible when it has at least four rows. Exactly 33 repository
names must be eligible. Sort eligible names by
`SHA256("ci-repair-bed-v2|repo|" + repo_name)`, breaking a hash tie by literal
name. Allocate, in order:

- mechanics: 3 repositories;
- opportunity: 10 repositories;
- development: 8 repositories;
- confirmation: 8 repositories;
- reserve: 4 repositories.

Within each repository, sort rows by
`SHA256("ci-repair-bed-v2|row|" + id)`, breaking a hash tie by literal ID, and
retain at most 12. The expected retained latent-state counts are respectively
`16 / 69 / 58 / 71 / 40`. No owner-specific fork may cross a split because
grouping is by `repo_name`, not `repo_owner/repo_name`.

Public artifacts contain repository names, counts, and hashes of selected IDs,
but never selected IDs or source contents. Development, confirmation, and
reserve contents remain sealed throughout source opportunity and mechanics.

## Frozen Diagnostic Actions

The opportunity audit may open only `workflow` and `logs` for mechanics and
opportunity rows. Concatenate log records in dataset order, retaining their
public `name` and `step_name` fields. The eight actions are fixed:

1. `workflow_head`: first 40 nonempty workflow lines.
2. `log_head`: first 30 nonempty concatenated log lines.
3. `log_tail`: last 30 nonempty concatenated log lines.
4. `first_error`: 21-line window centered on the first line matching
   `error|exception|fatal|failed|failure` case-insensitively.
5. `last_error`: 21-line window centered on the last matching line.
6. `traceback`: 31-line window centered on the first line matching
   `traceback|stack trace|caused by` case-insensitively.
7. `test_signal`: first 20 lines matching
   `test|assert|expected|actual|passed|failed` case-insensitively.
8. `dependency_signal`: first 20 lines matching
   `dependency|package|module|import|version|install|resolve` case-insensitively.

An absent match returns the literal `<NO_SIGNAL>`. Every response includes only
the selected view, not commit IDs or dataset row IDs. Before source partitioning,
strip ANSI escapes, lowercase, collapse whitespace, replace URLs, 7--64 digit
hex strings, UUIDs, ISO-like timestamps, standalone decimal numbers, and home or
runner absolute-path prefixes with typed placeholders. Preserve filenames,
extensions, exception names, command names, and ordinary semantic words. Hash
the resulting canonical response; raw excerpts are never public.

## Exact Non-Myopic Test

For each repository, use a uniform prior over its retained latent states. The
response model is deterministic. Compute action mutual information exactly from
the induced response partitions.

- greedy first action maximizes one-step mutual information, with action-ID
  lexical tie-breaking;
- its score is its expected two-step information after choosing the optimal
  second unused action separately in each realized branch;
- depth two chooses the first action maximizing that same expected two-step
  information, with the same tie-breaking;
- horizon gain is depth-two score minus the greedy-first score.

The source opportunity gate passes only if all of the following hold on the ten
opportunity repositories:

- at least 8 have four or more nonconstant actions;
- at least 8 have maximum root information below 95% of prior entropy;
- at least 4 select a different depth-two first action than greedy;
- at least 4 have horizon gain at least 0.05 nats;
- mean horizon gain across all 10 is at least 0.03 nats;
- every gain is finite and nonnegative to numerical tolerance.

Thresholds are frozen before any selected source response. Failure closes this
environment construction; thresholds, actions, normalization, split, and seed
may not be revised after inspection.

## Later Semantic Gate

A source pass still requires a new prospective mechanics protocol. At minimum it
must establish candidate-repair semantic support, strict likelihood
normalization, answer obedience on both common and rare branches, calibrated
truth probability, meaningful root action variation, and positive ranking
fidelity against the exact source planner. Any paid policy comparison must use
paired latent states, compute-matched myopic and random controls, sealed repair
outcomes, and a frozen cost boundary. DeepSeek may receive text only.
