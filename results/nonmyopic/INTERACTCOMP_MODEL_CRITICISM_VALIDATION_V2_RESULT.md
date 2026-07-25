# InteractComp Model-Criticism Validation V2 Result

Date: 2026-07-25

## Outcome

**V2 failed closed on multilingual question punctuation before classification,
auxiliary support, hidden context, or endpoints. The V2 block is closed.**

The next fresh screen again enrolled six collapsed supports successfully. All 24
root responses were single-line questions of valid length. Four roots for one
Chinese-language task ended with the standard full-width question mark `？`.
The inherited parser accepted only ASCII `?` and rejected the first such root.

## Execution

| Stage | Completed calls |
|---|---:|
| Initial particles for 16 screen tasks | 128 |
| Four roots for six enrolled tasks | 24 |
| Current/auxiliary classifications | 0 |
| Auxiliary support and semantic validation | 0 |
| Responder, refresh, endpoint | 0 |
| **Total** | **152** |

Enrolled indices were `90, 133, 125, 105, 69, 67` (benchmark IDs
`91, 134, 126, 106, 70, 68`) with initial unique counts `2, 3, 2, 1, 4, 3`.
Thus both fresh screens found the required six collapsed supports without target
selection.

No question was regenerated, rewritten, translated, or normalized. No partial
score or endpoint was computed.

## Integrity And Next Interface

The checkpoint contains only initial particles, target-blind enrollment counts,
and roots. It contains no classification, auxiliary particle, semantic
judgment, true response, hidden context, refreshed support, target answer, or
endpoint.

V2 receives no same-block rerun. A distinct V3 may use the next untouched
manifest block and prospectively accept either ASCII `?` or full-width `？` as
the final character of an otherwise unchanged bounded one-line question. This
is a multilingual punctuation correction only; all scientific scores, controls,
calls, gates, and the V2 whitespace-only classification grammar remain frozen.

## Cost

- Preregistered commit: `20e6ffc`.
- Run ID: `interactcomp-model-criticism-validation-v2-20260725T114500Z`.
- Requests/attempts: `152/152`.
- Retries/reasoning tokens/forced exits: `0/0/0`.
- Cost: `$0.0959226`.
- Private raw SHA-256:
  `b16ff355583328d7933ecc39fa28f01f8ac7e88ab3bf2b88848f13b542159149`.
- Public failure SHA-256:
  `7727af17f8d25dab403e3ffbbd77581c885989779453d4dd729cc38cd86b2497`.
- Project-ledger spend after failure: `$86.47568266920753`.
- Monday local allowance remaining: `$14.667622149999886`.
- Authenticated OpenRouter remaining: `$43.927137884`, or `$18.927137884`
  above the protected `$25` reserve.
- OatML resources used: none.
