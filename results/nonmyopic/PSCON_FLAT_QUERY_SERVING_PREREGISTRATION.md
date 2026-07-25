# PSCon Flat Semantic-Query Serving Gate

## Purpose

Test a simpler target-free output interface after the PSCon semantic-tree V1 JSON
assignment failure. This gate contains no hidden target, responder, policy score,
trajectory, or scientific endpoint. It exists only to establish that
`openai/gpt-5.4-mini` can reliably emit useful semantic product partitions before a
distinct efficacy test is designed.

The closed V1 conversation-64937 efficacy run is not rerun. The same already-open
conversation and visible candidate titles are used here solely as target-free serving
inputs.

## Frozen Source And Model

- PSCon commit: `42eabef33bdc7207841290fdbf4309e1a8d960f9`
- English conversations SHA256:
  `219c54ebd94bceca302c3c10b9e9b3b3c0beeda40fa4480c57c7241151bfd49d`
- English product graph SHA256:
  `d7b9dacc8c83aacaa174bb9955ec4240531076f1ff155bc4ccd07d4e53a4293b`
- Already-open conversation: `64937`
- Support: all 20 title-bearing products in its final search pool
- Seed: `24386`
- Model: `openai/gpt-5.4-mini`, explicitly non-thinking
- Temperature: `.7`
- Calls: exactly 10 in one batch
- Concurrency: 10
- Cost cap: `$0.15`; projected cost: `$0.03`
- No repair, normalization, continuation, or reissue
- No OatML

## Frozen Grammar

Each response must contain exactly three nonempty lines:

```text
QUESTION: <one question ending in ? or ？>
OPTIONS: <option A> || <option B> || <option C>
ASSIGNMENTS: <exactly 20 A/B/C characters with no spaces>
```

The parser requires:

- three exact prefixes;
- one bounded question;
- three distinct bounded options;
- exactly one `A`, `B`, or `C` per candidate in source order;
- all three labels used.

Raw responses are checkpointed before parsing.

## Frozen Gates

Passage requires all of:

- exactly 10 physical requests and 10 HTTP attempts;
- zero retries, reasoning tokens, and forced exits;
- all 10 responses parse;
- at least 8 unique questions;
- at least 5 unique assignment signatures;
- every partition has at least `.30` nats of entropy;
- generated partition entropy range is at least `.10` nats;
- total cost is at most `$0.15`.

Failure closes this exact flat Mini interface; there is no V3, parser broadening,
response repair, or rerun.

Passage authorizes only a separately frozen efficacy design on a different English
development conversation. It does not revive the closed V1 endpoint, establish
non-myopic value, or authorize Chinese confirmation.

## Dry Verification

Before any real response:

- 11 focused PSCon tests passed;
- Python compilation and `git diff --check` passed;
- the full deterministic 10-call source-backed fixture passed every frozen gate,
  with 10 unique questions, 10 unique partitions, and `.6543` nats of entropy range.
