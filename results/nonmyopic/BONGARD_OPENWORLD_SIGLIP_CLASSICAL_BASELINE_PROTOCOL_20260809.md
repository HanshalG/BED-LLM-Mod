# Bongard OpenWorld SigLIP Classical-Baseline Protocol

Frozen: 2026-08-09 (Europe/London), before any Bongard mechanics,
development, or confirmation endpoint was opened.

Status: **complete endpoint-sealed plan bank; zero API calls, zero candidate
outcomes, and zero endpoint outcomes**.

This protocol adds a stronger fixed semantic-vision comparator without
changing any frozen Luna request, task, action, endpoint, science gate, budget,
or headline rule. It also leaves the previously frozen DINOv2 comparator
unchanged. Neither classical baseline can authorize a paid call or endpoint
opening.

## Why This Control Exists

DINOv2 tests whether ordinary self-supervised visual features can support the
same two-query endpoint objective, but it does not fully answer whether
image-text pretraining already supplies enough semantic structure for this
task. SigLIP 2 is therefore a deliberately harder non-generative control. The
official model card describes improved semantic understanding, retrieval, and
VLM-transfer features, and the model is Apache-2.0:

- https://huggingface.co/google/siglip2-so400m-patch16-512
- https://arxiv.org/abs/2502.14786

This is not a contamination cure. Bongard-OpenWorld images are public and the
SigLIP 2 training mixture is web-scale, so training exposure cannot be ruled
out for either SigLIP or Luna. The control asks the narrower empirical question:
does a fixed pretrained semantic representation plus a deterministic
prototype update match the registered LLM belief machinery on the frozen
endpoint task?

## Frozen Representation

- Model: `google/siglip2-so400m-patch16-512`, revision
  `ceea1cba8130d8271436da4828633198c176a775`.
- Model-weights SHA-256:
  `a621bd212e1b3329b428595f9693217e19587afe826adf3e5c241a16392e8973`.
- Apache-2.0 vision tower: 428,772,800 parameters; normalized
  1,152-dimensional pooled image embedding.
- Runtime: PyTorch `2.10.0`, Transformers `4.57.5`, Apple MPS.
- Extraction is local and offline over all 2,296 frozen images. The executable
  recomputes model, config, and preprocessor hashes before loading weights.
- A resumable memmap binds model identity, revision, all model-file hashes,
  opaque image-row hash, shape, and batch size before accepting prior rows.
- The final index contains only partition, opaque task/image IDs, and
  source-byte hashes. It stores no role, candidate label, or endpoint label.

Frozen embedding artifacts:

- extractor manifest SHA-256:
  `9c3aa45197730421697c4eaa8cff6d395beeda624b188665af53f24dbd7946ba`;
- index SHA-256:
  `9f3a1de76dc7124c0a1a8872949b52e46fb7b85a5ee26267032aa548516e2733`;
- embedding array SHA-256:
  `338151bc734b9ea4259a57ccabacd6a499df5fe8afcf8cefdcdd3a4c6b13f4d1`.

The index hash is intentionally identical to DINOv2's: both encoders consume
the exact same endpoint-sealed image rows in the same order.

## Fixed Predictive Model

The predictive model is the same class-symmetric prototype likelihood used by
the DINO control. At a history, unit-normalized positive and negative
prototypes are normalized class means. For image embedding `x`:

```text
s(x) = cosine(x, positive prototype) - cosine(x, negative prototype)
P(positive | x, history) = sigmoid(scale * s(x)).
```

The bias is fixed at zero. Under global class complementation, `s` changes
sign and the predicted probability becomes its exact complement for any
signed scale.

Scale selection uses only 272 leave-one-out predictions among the four
initially observed images in mechanics plus development. No candidate or
endpoint outcome enters calibration. A first diagnostic inherited DINO's
positive-only grid and selected its floor `0.25`, whose LOO log loss
`0.695064` was worse than zero-scale `0.693147`. Before this protocol and
before any endpoint access, the calibration grid was corrected to include
zero and matched negative temperatures. The final frozen choice is `-2`, with
LOO log loss `0.685660`. The positive-grid plan bank is retained only as an
audit artifact and must never be scored or reported as the registered
comparator:

- diagnostic manifest SHA-256:
  `ff4ec10c087cc549e615609b062c2ce18e3974a317cd1e9f18f5abe2b9da8d8c`;
- diagnostic plans SHA-256:
  `6faa226351caa2ecf82ac2cedd28cbdd040c4ce8a07f52273cdd14490ddac60e`.

## Policies And Frozen Plans

`siglip_myopic` selects the candidate with maximum expected reduction in the
sum of marginal entropies of the two sealed endpoint predictions. After the
first realized label, it recomputes the one-step score for query two.

`siglip_depth2` selects query one by exact two-step expectimax over both first
labels and the best remaining second query. Query two is one-step PIG because
one query remains. Both policies use deterministic lexicographic ties and the
same prototype update.

Every first-label and second-label branch, second action, and final endpoint
probability is frozen before outcomes. The registered final plan bank is:

- plan manifest SHA-256:
  `95c00d948ee77589995c3434e3eb01b0a908b496fe2d984b859dbae4032d057d`;
- plans SHA-256:
  `a41cc3b01d18fa9008f67d6f60a3f113b8f3194b73e932b4e2c3cfc83215f587`.

Depth two changes a robust first action on 0/4 mechanics, 17/64 development,
and 21/96 confirmation tasks. This is planning opportunity, not efficacy.

## Outcome Interpretation

The frozen classical-suite adapter may load labels only after independently
replaying the existing Luna stage authorization. It follows each precomputed
realized branch and reports Brier score, log loss, accuracy, and truth
probability for both SigLIP policies. The primary horizon contrast is paired
`siglip_depth2 - siglip_myopic`; every metric receives the fixed 20,000-draw
task bootstrap.

Every future Luna endpoint report must include both DINO and SigLIP, even when
either classical model matches or wins. In that case the manuscript must not
claim that the task itself requires an LLM. A valid narrower claim can still
concern answer-conditioned changes in Luna's predictive belief state if its
registered within-Luna controls pass.

## Bindings

- embedding extractor SHA-256:
  `234933a1b3540d29100095142d364425cf70063fd2f66ce83aeae7a25a80d645`;
- SigLIP planner/interpreter SHA-256:
  `4fd19b2e9fabca09ef3d20e31d163dd7133784e5992d1f9afdbd77e495e84211`;
- unchanged shared DINO planner SHA-256:
  `ab30cb7add964eecfdb85194bd591c5be3dc6f78cd6a54d07c46868740a8bbfc`;
- extractor tests SHA-256:
  `3948dbd1b35b5c73ef1e1bf6c0ba47deac92c5288c90d03862f493040028f7a0`;
- planner tests SHA-256:
  `5d786839c91ba56bcc41ef56b2d8c53ca9178c5664dece9c4ac4652b3d61cf81`.

This protocol authorizes no paid call, endpoint opening, model change, policy
change, claim, or result-dependent rerun.
