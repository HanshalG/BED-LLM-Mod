# MovieLens Semantic Lookahead Ranking v6 Result

Date: 2026-07-24

Status: failed decisively; no policy or depth run is authorized.

Four users were prospectively enrolled from the final 15-user screen. Exact frozen
initial responses were resumed after a floating-point parser repair, then the scorer
and 16 branches completed. Total accounting was exactly 66 requests, zero reasoning,
and `$0.32814070`.

| Metric | Required | Observed |
|---|---:|---:|
| Semantic-score Spearman vs negative branch NLL | >= .25 | **-.345** |
| Mean semantic top-1 regret | improve by .02 | **.141** |
| Mean immediate-EIG top-1 regret | control | **.063** |
| Mean seeded-random top-1 regret | control | **.137** |
| Semantic choice beats immediate EIG | >= 2/4 | **0/4** |

The semantic scorer selected branches 1, 0, 2, 2; oracle branches were 2, 2, 0, 1.
It was worse than immediate EIG and slightly worse than random. The v5 opportunity is
therefore real, but this single-pass verbal model-aware lookahead does not identify it
and in fact ranks it backwards. Close the scorer without prompt or threshold tuning.

Remaining balance: `$19.435438173`.
