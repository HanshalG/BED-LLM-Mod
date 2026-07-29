# Number Game Qwen Pooled-Support Serving Result

Run: `number-game-qwen-pooled-support-serving-smoke-20260729T093842Z`

Status: **all gates pass; fresh 32-tree pooled-support cohort authorized**.

## Result

The smoke made exactly ten accepted requests and ten HTTP attempts, with zero
retry/provider retry, reasoning tokens, or forced exits. All ten independently
seeded responses were valid strict JSON. Cost was `$0.0122528`.

Pooling increased support at every fixed history:

- per-draw valid counts:
  `24/21`, `16/18`, `21/14`, `5/11`, and `12/6`;
- second-draw novel contributions:
  `7, 11, 4, 8, 3`;
- pooled valid counts:
  `31, 27, 25, 13, 15`;
- merged first supports: `29, 37`;
- merged second supports: `25, 20`.

Every second draw adds at least two distinct executable extensions, and all
pooled and deployed support minima pass. The two weakest individual
conditioned draws contain only five and six valid hypotheses, while their
pools contain 13 and 15. This directly supports the preregistered motivation:
independent pooling reduces single-generation support collapse.

No item salvage or semantic repair was used.

## Consequence

The separately frozen fresh 32-tree cohort on seeds `63100..63555` is
authorized unchanged. The passing mechanics smoke contains no target concept
or efficacy endpoint.

Public `RESULT.json` SHA-256:
`f4c5371e9cbe80e4a344fd7cb649e7e02d883c00768456c472ee2474f60b880a`.

Private raw-response SHA-256:
`e940186c3da9ad5d659a90a1a87228c4efdae6629e5ffb2613f410c39edc612f`.
