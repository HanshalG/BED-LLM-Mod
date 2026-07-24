# ClinDiag Fixed Evidence-Slot Audit

Date: 2026-07-24

Status: **static construction passes; a fresh support-mechanics gate may be
preregistered.**

## Construction

Every selected patient exposes the same eight generic actions:

1. present illness;
2. prior medical history;
3. family/social context;
4. first examination slot;
5. first laboratory slot;
6. second laboratory slot;
7. first imaging slot;
8. first other-test slot.

Procedure names and findings are hidden until their generic slot is selected. The
observation is copied from the pinned ClinDiag archive; no patient simulator or LLM
creates deployed evidence. Cases are eligible only when all eight slots are nonempty.
Biopsy, pathology, genetics, sequencing, treatment, surgery, transplantation, and
other confirmatory/intervention entries are filtered before assigning test slots.

## Result

After excluding all 114 prior ClinDiag case IDs, including the sealed 60-case holdout,
and applying the existing no-lexical-target-leak case filter:

| Subset | Complete fresh cases |
|---|---:|
| Challenging | 416 |
| Rare | 36 |
| **Total** | **452** |

The limiting slots were `other_1` (421 otherwise eligible cases missing), `lab_2`
(387), `lab_1` (200), `imaging_1` (179), and `exam_1` (134). This still leaves ample
fresh development and reserve capacity.

## Validity

This route fixes the two load-bearing environment defects:

- unlike retrospective procedure cards, every policy sees the same generic action
  labels and cannot infer the target from which named tests are available;
- unlike the gatekeeper routes, realized observations are deterministic stored data
  and cannot hallucinate or change under an exact replay.

The action contents remain retrospective case-report evidence, so this is not yet a
validated likelihood model. The next gate must separately test whether the LLM can:

1. regenerate stable open-world supports after individual stored chunks;
2. assign semantic likelihoods that predict held-out stored chunks under the true
   diagnosis better than distractor hypotheses;
3. exhibit a truth-anchored two-step opportunity before any policy comparison.
