# iCRAFT-MD Diagnosis-Profile Gates: Final Report

Date: 2026-07-14

## Protocol

This report closes the gate-only protocol registered in `PREREGISTRATION.md`.
It uses the hash-pinned MediQ iCRAFT-MD release at commit
`faa2ce62fef0423e35af4c31d7537aad973173eb`; the release SHA-256 is
`658441e6c6692c84d78fdf1c5ddb406ee6705f4a43330f0f759a1b941d8bfcdc`.
The target is the released `answer_idx`, which is the field scored by the
official benchmark. The release's two raw answer-text/index disagreements
(source IDs 112 and 129) are preserved and surfaced in manifests.

## Gate 1: Cost Micro-Smoke

**Passed.** On preregistered source ID 125, the non-thinking 26B A4B profile
scaffold generated exactly three fixed profiles for each of four diagnosis labels,
two validated root candidates, and two canonical `12 x 3` profile likelihood
tables. It used 50 requests, 13,927 prompt tokens, 3,297 completion tokens, no
reasoning tokens or forced exits, and $0.00272729, below the $0.05 ceiling.
The reproducible artifact is `SMOKE_REPORT.json`.

## Calibration Gate

The first calibration process was interrupted by the local execution wrapper at
288 requests and $0.02535745 before it wrote a report. A detached recovery was
also reaped after three requests and $0.00012947. Neither partial process produced
metrics and neither is used as a gate result.

The exact foreground recovery used the same frozen calibration IDs, seed, model,
three-profiles-per-label support, temperatures, retry limit, and $0.50 hard cap.
It terminated with `MediQ profile generation failed after bounded repairs` while
constructing the fixed support. The failure is captured in
`CALIBRATION_RECOVERY_2_REPORT_FAILURE.json`. It used 296 requests, 100,904 prompt
tokens, 41,936 completion tokens, no reasoning tokens or forced exits, and
$0.02559221.

This is a **profile-support gate failure**. The model did not supply the required
fixed latent support on the held-out calibration partition, so the protocol did not
evaluate prior quality, likelihood calibration, FactSelect deployment equivalence,
ranking fidelity, or the structural depth-two value. No partial output is treated as
evidence for any of those downstream gates.

## Consequence

The preregistration specifies that any failed gate stops this path. Therefore there
will be no profile-prompt tuning, retry, policy run, depth run, structural-gap test,
or ranking-fidelity test on iCRAFT-MD under this project. The external non-myopic
empirical claim remains unestablished. The result strengthens the validation-first
boundary finding: a plausible diagnosis target does not by itself yield a stable,
LLM-supplied generative patient state.

Total iCRAFT gate spend, including operationally interrupted attempts, was
$0.05380642. The OpenRouter ledger moved from $16.81168851 to $16.86549493.
