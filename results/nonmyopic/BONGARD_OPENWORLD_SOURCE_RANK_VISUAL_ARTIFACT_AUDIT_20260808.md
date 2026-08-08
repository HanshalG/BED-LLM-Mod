# Bongard OpenWorld Source-Rank Visual-Artifact Audit

Date: 2026-08-08 (Europe/London)

Status: **pass; zero model calls and zero scientific endpoint access**.

This audit follows and binds the prompt-role blindness audit. It changes no
model, task, image, seed, action, endpoint, prompt, request count, science gate,
threshold, or budget. It does not authorize paid calls.

## Threat

The public prompt does not expose source paths or source positions, and opaque
ID suffixes do not predict candidate versus endpoint role. The private source
protocol nevertheless always holds out class index 6 as an endpoint. If that
source rank has a systematic visible artifact, the endpoint role might be
partly inferable from pixels even though the metadata interface is clean.

## Frozen Classifier

For each image, the audit extracts 21 low-level features:

- encoded byte size, width, height, and aspect ratio;
- grayscale mean, standard deviation, and three quantiles;
- per-channel RGB mean and standard deviation;
- saturation mean and standard deviation;
- gradient-edge mean, standard deviation, and upper quantile;
- grayscale histogram entropy.

It includes both each raw feature and its within-task standardized value, for
42 inputs total. The classifier is an L2-regularized logistic regression.
Regularization is selected from six fixed values using five task-grouped folds
over mechanics plus development only. Confirmation labels never enter feature
selection, scaling, regularization selection, or fitting.

## Held-Out Result

The cross-validation rule selects `L2 = 1000`, the strongest regularization in
the fixed grid. On all 960 unobserved confirmation images:

| Diagnostic | Visual-feature model | Constant 20% prior |
|---|---:|---:|
| AUC | 0.540988 | 0.500000 |
| Brier score | 0.159434 | 0.160000 |
| Log loss | 0.498608 | 0.500402 |

Predicted endpoint probabilities range only from `0.158705` to `0.249456`.
A forced top-two choice recovers 47 of 192 endpoints, giving 24.48% precision
and recall. This is weak discrimination rather than proof of exact
independence, but it is below every frozen concern threshold and does not
provide a useful low-level role detector.

## Interpretation

Together with the metadata audit, this closes the concrete fixed-last-slot and
simple source-rank-artifact threats without changing the unopened protocol.
The result is deliberately narrower than a contamination audit: handcrafted
statistics cannot rule out semantic recognition or memorization of public
Bongard-OpenWorld images by a pretrained VLM. It also says nothing about belief
quality, calibration, non-myopic planning, or endpoint efficacy.

Executable manifest:
`results/nonmyopic/bongard_openworld_source_rank_visual_artifact_audit/bongard-openworld-source-rank-visual-artifact-audit-20260808/MANIFEST.json`

Manifest SHA-256:
`76b21f0718d9e6a5021326f5373caa64c05cd5b2a8ddf6cef2b60b29a8c91811`
