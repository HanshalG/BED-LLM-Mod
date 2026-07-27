# LongVid Contrastive Prompt-Only Serving Smoke Result

## Decision

The exact synthetic serving smoke **passes every frozen gate**. This
authorizes only a separately frozen contrastive mechanics gate on new
development tasks.

Run ID:
`longvid-contrastive-prompt-smoke-20260727T183000Z`.

## Results

| Gate | Result |
|---|---:|
| Physical requests / HTTP attempts | 10 / 10 |
| Exact parses | 10 / 10 |
| Retries / reasoning / forced exits | 0 / 0 / 0 |
| Changed refreshed supports | 8 / 8 |
| Unique refreshed supports | 8 / 8 |
| Grounded anchors per refresh | 6, 6, 6, 6, 6, 6, 6, 5 |
| Contrastive rank | B, 79% confidence |
| Cost | `$0.0816975` |

The run used 9,225 prompt tokens and 3,909 completion tokens. All responses
finished through ordinary nonreasoning chat and passed the strict exact-field
parser without extraction, cleanup, repair, retry, or reissue.

## Scope

This proves only that GPT-5.4 can transport the new prompt-only flat semantic
support and contrastive rank interface reliably on a public synthetic
fixture. It does not show that:

- supports track true LongVid evidence;
- generated searches retrieve necessary clips;
- four-step beliefs rank paths better than one-step beliefs; or
- a non-myopic policy improves a held-out endpoint.

Those questions require the separately preregistered development mechanics
gate with delayed necessary-clip access.

## Accounting

- Adapter cost: `$0.0816975`.
- Conservative remaining balance:
  `$33.321552594` (pre-run live balance minus full local cost).
- Protected through Monday, 3 August 2026: `$25`.
- OatML/Slurm/cluster use: none.
- Public smoke artifact:
  `results/nonmyopic/longvid_contrastive_prompt_serving_smoke/longvid-contrastive-prompt-smoke-20260727T183000Z/SMOKE.json`.
