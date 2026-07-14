# iCRAFT-MD Diagnosis-Profile Gate Preregistration

Date: 2026-07-14

Commit at registration: `d5dc7f5`.

## Scope

This registers iCRAFT-MD diagnosis-profile validation gates only. It does not authorize
a policy comparison, a full-depth evaluation, or a retry after a failed gate. The initial
OpenRouter cap is `$0.50` per run. The 26B A4B scaffold is non-thinking.

## Immutable Contract

- Repository: `https://github.com/stellali7/MediQ.git`
- Commit: `faa2ce62fef0423e35af4c31d7537aad973173eb`
- Split: `data/all_craft_md.jsonl`
- SHA-256: `658441e6c6692c84d78fdf1c5ddb406ee6705f4a43330f0f759a1b941d8bfcdc`
- Target: released four-way diagnosis label.
- Patient channel: official Fact-Select only.
- Outcome support: `Yes`, `No`, and `Information unavailable / not in record`.

The fixed source-ID partition comes from `random.Random(1304).shuffle(range(140))`.

| Partition | Count | Source IDs |
|---|---:|---|
| Development | 8 | 125, 83, 124, 38, 8, 92, 104, 135 |
| Calibration | 12 | 2, 99, 132, 60, 62, 64, 23, 137, 96, 100, 117, 40 |
| Structural | 12 | 30, 52, 103, 33, 85, 36, 94, 113, 34, 139, 15, 68 |
| Pilot reserve | 10 | 93, 80, 21, 115, 138, 12, 108, 119, 131, 126 |
| Headline reserve | 98 | All remaining source IDs |

## Fixed Model

Each diagnosis label receives exactly three validated counterfactual patient profiles.
Profiles use only initial evidence and the diagnosis; they may not use hidden atomic facts
or assert treatment, management, or test-order decisions. The direct temperature-zero
diagnosis prior is divided uniformly among its three profiles. Profile support is fixed
after validation: it is never regenerated or refreshed in a hypothetical branch.

Record availability is label-independent. Conditional on availability, Yes/No is
profile-conditioned. The same likelihood table is used for EIG, synthetic branches, and
deployed Fact-Select Bayes updates.

## Gates

1. Local contract: routing tests and the full suite pass before paid calls.
2. Cost micro-smoke: source ID 125, one round, two shared candidates, no parse/runtime
   failure, and spend at most `$0.05`.
3. Profile/prior: all calibration labels retain three profiles; mean log loss is below
   `log(4)`; mean Brier is below `0.75`; at most one wrong top diagnosis has probability
   at least `0.75`.
4. Likelihood/deployment/ranking: four root candidates for each calibration case, with
   at least 24 available outcomes, unavailable diagnosis movement at most `1e-12`,
   positive available true-label log gain, at least 60% true-label favoring, and
   Spearman EIG correlations at least `0.20` with realized entropy and truth-log gain.
5. Structural: on the structural partition, exact depth-two value minus the one-step
   root's depth-two value has mean at least `0.02` nats, at least 8/12 positive cases,
   and a one-sided 90% paired-bootstrap lower bound above zero (10,000 replicates).

Any failed gate stops this path. A structural failure does not permit a FactSelect rescue
attempt. A structural pass still requires a second authorization before a policy run.
The banked animals audit is one-step evidence and a negative paired depth result, never
positive non-myopic evidence.
