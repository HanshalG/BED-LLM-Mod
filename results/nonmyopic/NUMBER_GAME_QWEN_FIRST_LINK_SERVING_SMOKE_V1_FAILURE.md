# Number Game Qwen First-Link Serving Smoke V1 Failure

Run: `number-game-qwen-first-link-serving-smoke-20260729T082609Z`

Status: **serving gate failed; confirmation not run**.

## Serving

The exact ten `qwen/qwen3.7-plus` requests completed cleanly:

- accepted requests / HTTP attempts: `10/10`;
- retries / provider-error retries: `0/0`;
- reasoning tokens / forced exits: `0/0`;
- cost: `$0.01082432`;
- all ten responses parsed;
- initial valid unique counts: `23,23`.

Conditioned valid unique counts were:

`18,21,5,19,10,7,18,24`.

The frozen requirement that every raw conditioned support contain at least
eight rules therefore failed on two cases. The five-rule case had 19
duplicate extensions; the seven-rule case had 16. No response was repaired
or reissued.

## Consequence

The 64-tree, 3,712-call confirmation was not started, and all its seeds and
canonical endpoints remain unopened.

V1 exposed an instrument mismatch: the actual policy uses retained
rejuvenation and gates the merged parent-plus-child support, whereas this
smoke gated the newly generated child support alone. V1 remains failed. A
successor would require a separately registered linked-support smoke and
cannot relabel or reuse V1 for authorization.

Public `RESULT.json` SHA-256:
`1afee4df0408eb1edbf013d7c8dfb997e99d1b9a9c520175210bff04b2f85618`.

Private raw-response SHA-256:
`8b946770965f613f8222411db9bf9b58f214cdb858c11614fb868d83b4854fd9`.
