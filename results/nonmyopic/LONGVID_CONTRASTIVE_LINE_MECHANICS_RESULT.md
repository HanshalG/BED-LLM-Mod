# LongVid Fixed-Line Contrastive Mechanics Result

## Decision

The final LongVid development interface **failed on an OpenRouter transport
truncation before the first refresh batch could be checkpointed**. Scientific
efficacy is unmeasured, the endpoint remained sealed, and the LongVid
contrastive cycle is closed.

Run ID:
`longvid-contrastive-line-mechanics-20260727T193000Z`.

## Progress Before Failure

The four initial six-line support requests completed and parsed. During the
first eight-way refresh batch, one HTTP response ended mid-chunk:

```text
IncompleteRead(242 bytes read)
```

The adapter had zero retries by preregistration, so the batch raised
immediately. Eleven requests were accepted across twelve HTTP attempts: four
initial calls and seven refresh calls. Because the complete refresh batch is
checkpointed atomically, no partial refresh set was admitted to the
scientific artifact.

## Accounting And Endpoint State

| Item | Result |
|---|---:|
| Accepted requests | 11 |
| HTTP attempts | 12 |
| Prompt / completion tokens | 6,993 / 3,843 |
| Reasoning tokens | 0 |
| Semantic retries / forced exits | 0 / 0 |
| Cost | `$0.0751275` |
| Complete initial support batch | 4 / 4 |
| Complete refresh batches | 0 / 4 |
| Immediate/final ranks | 0 / 0 |
| Endpoint loaded | no |

No retry, partial-batch acceptance, response recovery, rerun, or remaining-row
substitution is permitted.

## Interpretation

This is a network-transport failure, not evidence about:

- fixed-line grammar quality after the initial batch;
- path-dependent support value;
- one-step versus four-step ranking; or
- realized necessary-clip coverage.

It does reveal a protocol-design lesson. Treating every transient HTTP
failure as a scientific failure is unnecessarily brittle for long batched
LLM experiments. Future protocols should prospectively distinguish bounded
transport retries from semantic reissues while retaining raw response,
attempt, token, and cost accounting. That change cannot be applied
retrospectively here.

The replicated classical LongVid four-hop opportunity and successful
synthetic support smoke remain valid. No LLM ranking claim was measured.

## Closure And Budget

- All LongVid contrastive development interfaces in this cycle are closed.
- The two remaining strict confirmation rows cannot be used as a rescue.
- The untouched four-hop reserve remains unopened.
- Conservative live remaining OpenRouter balance: `$33.206015094`.
- Protected through Monday, 3 August 2026: `$25`.
- OatML/Slurm/cluster use: none.
- Private raw SHA-256:
  `b97792c84e663f97750016c9237765183f280768cbe1410c5799798dd84e2a90`.
