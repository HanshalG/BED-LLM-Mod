# DiscoverLLM Priority-World Mechanics Result

## Verdict

The exact mechanics interface **fails closed at final likelihood parsing**. All
five semantic stages and all 15 requests completed, but 45 of 96 final
likelihood values used forbidden zero-padded digit strings. No Bayesian score,
policy selection, hidden-world endpoint, or opportunity claim is computed.

Per the frozen protocol, there is no coercion, parser amendment, prompt change,
or format-only rerun. The exact fixed-world/two-action DiscoverLLM construction
is closed. The 60 opportunity, 30 development, and 162 holdout artifacts remain
unopened.

## Integrity

- Run:
  `discoverllm-priority-world-mechanics-20260726T004542Z`
- Model: `openai/gpt-5.4`, non-reasoning, temperature zero
- Exact requests/HTTP attempts: `15 / 15`
- Retries/reasoning tokens/forced exits: `0 / 0 / 0`
- Prompt/completion tokens: `75,616 / 5,728`
- Cost: `$0.274960`
- Private raw SHA-256:
  `813821a8a9c01a63a6ab585ec574314db43a0792d29c6da6a652a3e9ed84e366`

## Failure

Every stage returned the expected number of complete JSON objects:

| Likelihood stage | Objects | Keys | Canonical strings | Invalid strings |
| --- | ---: | ---: | ---: | ---: |
| Root | 3 | 96 | 96 | 0 |
| Follow-up | 3 | 96 | 51 | **45** |

All invalid values were decimal digit strings with a leading zero, such as
`"05"` or `"03"`. There were no non-string, nondigit, out-of-range, missing-key,
or malformed-JSON values. This is a serving-schema failure, not evidence for or
against a non-myopic opportunity.

The raw responses contain all five stages, but the preregistration forbids
coercion and endpoint calculation after any parse failure. They are retained
only as a private audit artifact.

## Budget

After the run:

- authenticated provider remaining: `$37.668924994`;
- remaining above the protected `$25` reserve: `$12.668924994`;
- stricter project-ledger headroom: `$12.060764791`.

OpenRouter only. OatML jobs: `0`.
