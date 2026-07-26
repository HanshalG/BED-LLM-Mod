# VoI Medical Dynamic-Support Manifest Preregistration

## Objective

The fixed 15-diagnosis V2 tree was coherent but myopic and depth-two EIG chose
the same root. The next route targets the headline mechanism directly:
path-dependent LLM-generated clinical hypotheses whose truth coverage can
change after a hypothetical answer.

Before opening patient text for this route, freeze a value-blind row split over
the official MedDG source.

## Frozen Source And Split

- Official repository commit:
  `27466a7832d5aafff82017a659e08942e18b01ae`
- Source: `mixed_20q/data/MedDG.json`
- Source SHA-256:
  `e851864a9cb53c36304245bc3213a8a894cf7b86f8945d60978923e1f1ef0169`
- Rows: `499`
- Seed: `24423`
- Shuffle: Python `random.Random(seed).shuffle(range(499))`
- Ordered slices:
  - mechanics: first `5`
  - opportunity: next `40`
  - development: next `20`
  - holdout: remaining `434`

The manifest may validate row count, exact keys, string types, and source hash.
It emits only row indices, counts, split hashes, and protocol metadata. It does
not inspect or emit self-report, conversation, or target values.

## Access Policy

1. Generate and commit the manifest before opening any split values.
2. Open only the five mechanics rows to audit answerability and construct the
   dynamic-support instrument.
3. The opportunity block may be opened only under a separately frozen,
   target-blind scoring protocol.
4. Development and holdout remain sealed until their preceding gates pass.

The policy generator will not receive the released 15-label candidate list.
It must generate free-form clinical hypotheses from visible patient evidence
and complete hypothetical question-answer histories. The external diagnosis is
loaded only after policy scores are frozen and is used solely for truth-coverage
and terminal diagnostic endpoints.

No OpenRouter calls or OatML jobs are authorized by this manifest alone.
