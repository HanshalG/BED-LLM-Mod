# Number Game Depth-Three Uncertainty Gate-128 Result

Run: `number-game-depth-three-uncertainty-gate128-20260729T081711Z`

Status: **retrospective uncertainty-gate null**. This analysis does not alter
any source study's registered status.

## Gate Decisions

The fixed target-blind rule saw 52 shared depth-two/depth-three roots and 76
candidate root changes. It:

- accepted 55 depth-three changes;
- fell back to depth two on 21 changes;
- recomputed all eight-draw mean advantages to within `2.43e-17` of the
  source-study values.

## Exact-Canonical Endpoint

Against depth two across all 128 trees, the gated policy has:

- Brier `0.107035` versus `0.108449`;
- relative reduction `1.304%`;
- mean difference `-0.001414`;
- four-block stratified interval `[-0.003077,+0.000147]`;
- wins/ties/losses `29/73/26`.

The point estimate remains favorable, but the interval crosses zero. The
planner-family split is heterogeneous:

- Qwen 3.7 Plus: `2.426%`, interval
  `[-0.005196,-0.000260]`;
- GPT-5.4 Mini: `0.200%`, interval
  `[-0.002339,+0.001787]`.

Among the 55 accepted changes, predicted-versus-realized advantage Spearman
is only `0.107`.

The gate preserves the non-myopic-over-myopic result: Brier improves by
`12.019%`, interval `[-0.018685,-0.010830]`, with 93/128 wins.

## Diagnostic

Post hoc arithmetic over the frozen per-tree rows shows why the gate does not
help. Accepted depth-three changes improve on depth two by `0.003292` on
average with 29/26 wins/losses. Rejected changes improve by `0.003391` with
15/6 wins/losses. Stability across validation supports therefore does not
separate useful from harmful deeper root changes in this sample.

The fixed gate is closed without tuning its six-of-eight or one-standard-error
thresholds. The evidence still supports depth three over myopic EIG, but not a
monotonic depth-three-over-depth-two claim.

Model calls and cost are zero. Public `RESULT.json` SHA-256:
`bcdf37f746b156f0b849f5b1a94b4b4a2f31348bd146260a92a0358a2ea92bae`.
