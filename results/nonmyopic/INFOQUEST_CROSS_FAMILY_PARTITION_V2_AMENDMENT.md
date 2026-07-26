# InfoQuest Cross-Family Partition V2 Budget Amendment

Frozen after V1 failed local budget preflight and before any cross-family model
response or endpoint.

V1 made zero physical requests and zero HTTP attempts, used zero tokens, and
cost `$0`. Its launcher set projected run spend to `$0.08` for both stages,
which exceeded the separately frozen `$0.03` serving cap. The failure occurred
while constructing adapters, before prompts or private raw files existed.

V1 is closed. V2 changes only:

- interface version from `infoquest-cross-family-partition-1` to
  `infoquest-cross-family-partition-2`;
- serving projected-spend hint from `$0.08` to `$0.02`.

The serving hard cap remains `$0.03`; mechanics projected spend remains `$0.08`
and its hard cap remains `$0.15`. Every source hash, cached support/history,
model, non-reasoning setting, prompt, parser, exact scorer, action bank,
cached-answer endpoint, call count, science threshold, no-repair rule, and
fresh-confirmation requirement remains byte-for-byte or semantically
unchanged.

The pre-Monday operational allowance remains `$2.08299670`; V1 spent `$0` and
V2 can consume at most `$0.18`. OatML jobs: `0`.
