# GuessWhat?! Semantic BED Serving Smoke Result

Date: 2026-07-29

Status: **semantic gates failed; the exact interface is closed**.

## Bound Protocol

The run used the two frozen serving images from source manifest SHA-256
`027a7f49fc599001eca6fbac0b3fe3e6be021d8cb3512eacad3ee167d0f5fff2`.
The implementation and preregistration were committed and pushed as
`4c598e0` before any live response.

- GPT-5.4-Mini proposed four visual binary questions per image.
- Gemini 2.5 Flash assigned candidate-conditioned `Yes` probabilities.
- Qwen3-VL-32B-Instruct independently answered Q1 through Q3 about the
  released target overlay.
- All models were non-thinking, and no retry, repair, reissue, model swap,
  or threshold change was permitted.

## Result

Transport and parsing were clean:

| Metric | Result |
|---|---:|
| accepted requests / expected | 10 / 10 |
| HTTP attempts / expected | 10 / 10 |
| schemas parsed | 10 / 10 |
| retries | 0 |
| provider-error retries | 0 |
| reasoning tokens | 0 |
| forced exits | 0 |
| cost | `$0.005813914` |

The semantic conjunction failed:

| Frozen gate | Result | Required |
|---|---:|---:|
| unique semantic questions | 4 / case | 4 / case |
| discriminatory questions | 0 / 8 | >=2 / case |
| balanced questions | 3 / 8 | >=2 / case |
| cross-model consistency | 4 / 6 | >=5 / 6 |
| minimum per-case consistency | 1 / 3 | >=2 / 3 |
| oracle answer coverage | Yes and No | Yes and No |

Gemini emitted only two distinct probability values on seven question rows
and one value on the eighth. Six rows were exact `0/100` partitions. They
could have high range, but none met the prospectively frozen requirement for
at least three distinct probability values. Their predicted `Yes` mass was
also too imbalanced on one image. On that image, Qwen answered `Yes` to two
questions for which Gemini assigned the target at most 40%, producing only
one of three consistent target answers.

## Interpretation

The visual planner produced syntactically valid and distinct semantic
questions, but this tested independent visual-likelihood factorization did
not produce a calibrated enough response model for BED. The failure is not a
transport artifact. First-link ranking fidelity, policy evaluation, and all
development and holdout images remain unopened.

There is no rerun or repair of this exact prompt/model/case/seed/threshold
interface.

## Artifacts

- Public result:
  `results/nonmyopic/guesswhat_semantic_bed_serving_smoke/guesswhat-semantic-serving-20260729T014832Z/SERVING.json`
- Public result SHA-256:
  `a74a6d96f3fd85e2e699a68bdde0702c109277930d6018d578a1712c41591d0e`
- Private raw-response SHA-256:
  `ef228fd2152e21950714cb964d7cf96551b912e7c93c49f9dce2fc80fed68932`

`run.log`, `console.log`, images, overlays, and private raw responses are not
committed.
