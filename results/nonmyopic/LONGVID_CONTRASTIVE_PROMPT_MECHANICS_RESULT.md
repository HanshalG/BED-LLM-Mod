# LongVid Contrastive Prompt-Only Mechanics Result

## Decision

The prompt-only JSON mechanics gate **failed before path execution and
scientific scoring**. The exact method and its four rows are closed.

Run ID:
`longvid-contrastive-prompt-mechanics-20260727T190000Z`.

## Failure

All four initial-support requests completed through GPT-5.4 ordinary chat,
but none returned the required single flat JSON object. Each response emitted
six comma-separated mini-objects. Strict `json.loads` therefore stopped on
extra data after the first four-field object.

The first error was:

```text
JSONDecodeError: Extra data: line 1 column 329 (char 328)
```

No first object was extracted, no surrounding objects were collected into an
array, and no response was repaired or reissued.

## Accounting And Endpoint State

| Item | Result |
|---|---:|
| Requests / HTTP attempts | 4 / 4 |
| Prompt / completion tokens | 1,531 / 1,922 |
| Reasoning tokens | 0 |
| Retries / forced exits | 0 / 0 |
| Cost | `$0.0326575` |
| Root searches executed | 0 |
| Model trajectory captions | 0 |
| Endpoint loaded | no |

The four responses were checkpointed before parsing. The public failure
artifact records private raw SHA-256
`02999ceee8e726aff0437c937de720e974f41f36a9f7d0d8b4fdc85b657571ee`.

## Interpretation

The synthetic prompt-only smoke passed, but the same large flat-object
instruction did not generalize to real LongVid questions. This is a serving
robustness failure, not evidence for or against contrastive non-myopic
ranking.

Do not rerun rows `1404`, `2703`, `1867`, or `1295`, collect the mini-objects,
change the parser, or alter the threshold/model inside this protocol.

A materially different fixed-line contrastive method may use new tasks
because the earlier real-task entropy run established that six-line supports
were syntactically stable; its failure concerned an ASCII-only anchor
validator, not line transport. Such a method requires a new preregistration
and cannot rescue this result.

## Budget

- Local cost: `$0.0326575`.
- Conservative remaining balance:
  `$33.288895094` (prior conservative balance minus local cost).
- Protected through Monday, 3 August 2026: `$25`.
- OatML/Slurm/cluster use: none.
