# Bamboogle Semantic-BED Manifest Result

## Decision

The content-blind manifest reproduces exactly. The five mechanics records are
released for interface development; the 20 opportunity, 20 development, and
80 holdout questions and answers remain sealed.

## Reproduction

- Source rows: `125`.
- Source schema: exactly `id`, `question`, and `golden_answers`.
- Source SHA-256:
  `c9703dae6bb1ceb9e2df77be45da28cb12aa040d2f471507890a461296968f3f`.
- Selection seed: `24404`.
- Public manifest SHA-256:
  `08b893262d4824b48c0783f0fdc06b907bcfaf8e1a35df6a28aa21c7e486a2e6`.
- Question or answer content emitted by the manifest: `false`.

| Split | Size | Ordered ID SHA-256 |
| --- | ---: | --- |
| Mechanics | 5 | `f2a9e1d7587e22d16127dae8ed15dd8c58fcc9b867ce1a1a1fc13f9e01912b7e` |
| Opportunity | 20 | `6abfc7dc790564fa09cc6fb282851beece36306c682e038e177e39b546aa2e94` |
| Development | 20 | `01837e7e950110c5da05440862d443092a1edc9ce14b0032c8942d5bbfedb8a6` |
| Holdout | 80 | `c2da4d412ae4d7c14c3aca06033f61ba5ec1078789ab70a39811eeeed3f85856` |

## Mechanics Scope

The released mechanics IDs are `test_87`, `test_110`, `test_72`, `test_69`,
and `test_61`. They are compositional questions whose answers require resolving
an intermediate entity or ranking. Several facts are common enough that a
frontier model may answer from memory, so no-search saturation is a
load-bearing stop gate.

No search transition, generated belief, policy score, or endpoint comparison
has been run. There were no OpenRouter calls and no OatML jobs.
