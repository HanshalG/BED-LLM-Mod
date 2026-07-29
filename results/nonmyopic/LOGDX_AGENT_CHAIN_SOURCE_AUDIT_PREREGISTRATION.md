# LogDx-CI Agent-Chain Source Audit Preregistration

Date: 2026-07-29

## Scientific Question

Can LogDx-CI support a genuinely non-myopic, LLM-native BED experiment in
which an LLM maintains semantic root-cause hypotheses and chooses an evidence
query partly because its observation enables a better second query?

The relevant mechanism is an observation-contingent tool chain, not simply
allowing more tool calls. A broad first query may expose a file, line number,
test name, or error token that makes a targeted second query possible. The
terminal endpoint is diagnosis quality against the released root-cause
annotation.

## Pinned Source

- paper: `https://arxiv.org/abs/2605.28876`;
- official repository: `https://github.com/eyuansu62/LogDx`;
- release tag object: `8d358ae6b973320b8a5e29ce7e0eab243033d31f`;
- checked-out `v1.2` commit:
  `99591c1471118c95155976346df72f520a05f100`;
- protocol lock SHA-256:
  `c5e77e91267036ceb1e80df1d035caa78eb0c831ae690ab5eb0149215fb36087`;
- deterministic tool implementation SHA-256:
  `4c839cc40ad90f530fe08cb94daa020e69de72eec6b575dbcaa009bb0a71afcc`;
- diagnosis evaluator SHA-256:
  `ff6a51b19049b40c80ad29c7c9ca3a25fc184b01d443acbcc9c3463b12472e54`.

The six committed split manifests contain `35` cases in total:

- legacy `dev`, `holdout`, and `stress`: `5`, `5`, and `6`;
- v2 `dev`, `holdout`, and `stress`: `3`, `10`, and `6`.

No raw log, ground-truth diagnosis, model diagnosis, or agent trace may be
opened before this preregistration is committed.

## Source And Execution Gates

The source passes only if all of the following hold:

1. the pinned protocol validator and all six split-manifest content hashes
   validate locally;
2. `grep`, `read_file`, `tail`, and `view_log_lines` are deterministic
   operations over only the case's `raw.log`;
3. the released agent prompt cannot access `ground_truth.json`, evaluator
   outputs, or unrestricted filesystem tools;
4. diagnosis scoring is deterministic for a fixed diagnosis and released
   ground truth;
5. case annotations specify a concrete root cause and evidence rather than
   only a coarse failure category.

Any source or leakage failure closes this route before model calls.

## Zero-Call Published-Trace Gate

The audit will use only already released `real-agent-v1` and matched
`real-debugger-v2` artifacts. It will not rerun either model. For every
case-context pair with both artifacts, it will record:

- tool-call count and ordered tool names;
- diagnosis score for the agent and matched single-shot debugger;
- score difference;
- whether a later tool call is observation-dependent;
- the dependency type: discovered line number, file/path, test or symbol,
  error token, or other literal search term.

A later call counts as observation-dependent only when one of its substantive
arguments is absent from the assistant-visible initial context and appears
literally in an earlier tool observation. Generic constants such as
`error`, `failed`, line `1`, or a fixed tail length do not count. Dependency
classification must be implemented mechanically and accompanied by a
redacted case-level table containing IDs, tool names, scores, and dependency
types but no raw log or root-cause text.

The published-trace gate passes only if all conditions are met:

- at least `25` distinct cases have matched artifacts;
- at least `12` distinct cases use one or more tools;
- at least `8` distinct cases use two or more tools;
- at least `6` distinct cases contain an observation-dependent later call;
- at least `5` distinct cases both contain such a dependency and improve
  diagnosis score over matched single-shot by at least `.10`;
- mean paired score gain over all matched rows is positive;
- mean paired score gain over dependency rows is at least `.10`;
- at least two dependency types occur.

The unit of support for count gates is the distinct case, not repeated context
providers. For score means, first average within case across matched context
providers, then average cases.

This is an opportunity audit, not a policy comparison. Passing it authorizes
only a separately frozen implementation and ten-call serving test.

## Conditional Method

If the full conjunction passes, the first implementation will use:

- the same deterministic LogDx tools and released diagnosis endpoint;
- a small assistant-visible initial context such as `tail-200`;
- LLM-generated root-cause hypotheses and calibrated semantic likelihoods;
- matched two-query budgets for all policies;
- a receding one-step EIG baseline;
- a depth-two contingent StrategyEIG policy;
- a compute-matched depth-two policy scored only by first-step EIG;
- a random valid-query control;
- common cases and deterministic tool observations.

The first-link gate will measure whether predicted strategy value ranks
realized truth-anchored diagnosis improvement before any powered policy claim.
Thinking may be used for the naive diagnostic baseline, but the BED policies
start nonthinking unless the serving gate demonstrates that a thinking model
is needed for valid semantic objects.

## Cost Boundary

This audit makes zero model calls and spends `$0`. OpenRouter is the only
permitted paid backend after a pass. The reported `$40` top-up is not assumed
available until it appears in the authenticated credits endpoint. There is no
protected reserve, but paid work remains conditional on the frozen gates.
