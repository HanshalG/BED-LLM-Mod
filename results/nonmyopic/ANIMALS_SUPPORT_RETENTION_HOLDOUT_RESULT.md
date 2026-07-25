# Animals Support-Retention Holdout Result

Date: 2026-07-25

## Verdict

The preregistered support-retention holdout failed closed during the eighth
of 60 states. Seven states completed, but there is no aggregate scientific
endpoint and the partial records are not analyzed for efficacy.

The failure occurred while parsing a target-blind semantic classification:

```text
JSONDecodeError: Expecting ':' delimiter: line 1 column 9811 (char 9810)
```

The provider reported `finish_reason=stop` for this response and for every
other response in the run. The failing response used 1,142 completion tokens;
the run maximum was 3,369, below the frozen 16,384-token cap. This was
therefore malformed model output rather than token truncation.

## Execution

- Run:
  `animals-support-retention-holdout-20260725T224525Z`.
- Model: `google/gemma-4-26b-a4b-it`, non-thinking.
- Completed states: `7 / 60`.
- Requests / HTTP attempts: `3,981 / 3,981`.
- Prompt / completion tokens: `516,674 / 160,532`.
- Retries / reasoning tokens / forced exits: `0 / 0 / 0`.
- Cost: `$0.11221393`.
- Public failure SHA-256:
  `18732dfefdfd883ea9f765610df03d6e82a6792b2f510d27f9aacd7685b450ba`.
- Private checkpoint SHA-256:
  `de846b081d4926f0ffa79300f8add71ebd88635d027ae7562d8e6147113d1c08`.

The private checkpoint contains 56 generation records and 110 successfully
parsed semantic-classification records. The recorder appends a semantic
classification only after parsing succeeds, so it does not contain the
malformed raw response itself. The traceback, parse position, token-usage
event, and provider finish reason are retained, but the raw response is not.
This is an explicit forensic limitation of the artifact.

## Decision

The exact support-retention interface is closed without parser repair,
response recovery, target subsetting, or rerun. The seven partial records
cannot be used to estimate, select, or describe selector efficacy.

The post-hoc retention pattern in development remains only a hypothesis. The
failed support-expansion development remains null, and the preregistered
multi-round policy is not authorized.

Immediately after the failure, the authenticated OpenRouter balance was
`$38.483769494`, leaving `$13.483769494` above the protected `$25` reserve.
The stricter local ledger allowed `$13.098966791`. No OatML resources were
used.
