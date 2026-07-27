# LongVid Contrastive Path-Belief Mechanics Result

## Decision

The strict-schema interface **failed at provider routing before any model
response**. Scientific efficacy is unmeasured and this exact four-task
interface is closed.

Run ID:
`longvid-contrastive-path-belief-20260727T180000Z`.

## Failure

The first batch attempted four initial semantic-support requests. OpenRouter
returned HTTP 404 for each:

```text
No endpoints found that can handle the requested parameters.
```

The failure occurred because no routed GPT-5.4 provider accepted the strict
JSON-schema payload. It was not a parser error, model refusal, or malformed
support.

Exact accounting:

| Item | Result |
|---|---:|
| HTTP attempts | 4 |
| Accepted physical requests | 0 |
| Responses | 0 |
| Prompt/completion/reasoning tokens | 0 / 0 / 0 |
| Retries / forced exits | 0 / 0 |
| Adapter cost | `$0.00` |

No root query was executed, no LongVid caption was retrieved for a model
trajectory, no policy checkpoint was produced, and the necessary-clip
endpoint remained unloaded.

## Consequence

Do not rerun rows `2156`, `2062`, `1689`, or `1648`, remove strictness, swap
models, or fall back inside this interface. A future protocol must use new
development tasks and pass a separately frozen synthetic serving smoke before
scientific execution.

The contrastive scientific question remains open: whether four-step
LLM-regenerated semantic beliefs rank realized evidence-chain coverage better
than one-step beliefs.

## Accounting

- Live remaining OpenRouter balance after failure: `$33.403250094`.
- Protected through Monday, 3 August 2026: `$25`.
- OpenRouter charge: `$0`.
- OatML/Slurm/cluster use: none.
- Public failure artifact:
  `results/nonmyopic/longvid_contrastive_path_belief_mechanics/longvid-contrastive-path-belief-20260727T180000Z/MECHANICS_FAILURE.json`.
- Private raw checkpoint SHA-256:
  `2eb4551b5859ae5392a3c58a8e7521a80ac49788e24586a696f03e8b5db9635a`.
