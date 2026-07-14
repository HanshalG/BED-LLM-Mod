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

## Authorized Stronger-Generator Retry

After the independent animals re-analysis collapsed, Hanshal authorized exactly one
separate apparatus retry. It changed only profile narrative authoring to
`openai/gpt-5.4` with reasoning disabled. The non-thinking 26B model remained the
profile validator, prior judge, candidate generator/validator, likelihood scorer,
FactSelect patient, mapper, and every policy role. Prompts, calibration IDs, support
size, repair limit, temperatures, seed, and thresholds were unchanged.

**Profile construction passed, calibration failed.** The retry constructed the fixed
profile supports and reached every calibration metric. Its grouped diagnosis prior met
the registered quality criteria (mean log loss `0.56755`, threshold `< log(4)`; mean
Brier `0.28125`, threshold `< 0.75`). The decisive failure was the official channel:
only 14 of 48 realized candidate observations were available, below the preregistered
minimum of 24. No amount of post-hoc likelihood or planning analysis can repair that
missing interaction signal.

For transparency, the raw run also reported negative EIG correlations with realized
entropy (`-0.04131`) and truth-log gain (`-0.06777`). A post-run code audit found that
the posterior probability floor perturbed low-mass profile particles after an
otherwise label-independent unavailable outcome; the corrected code now regression-tests
exact unavailable neutrality. Those raw rank numbers are therefore retained as forensic
telemetry rather than an independent gate conclusion. The availability failure is
independent of that correction and is sufficient to fail the gate.

The run used 1,000 requests, 321,398 prompt tokens, 57,925 completion tokens, zero
reasoning tokens, and zero forced exits for `$0.44032225`, below its `$2` hard cap.
Cost attribution was `$0.39681000` for 55 GPT-5.4 authoring calls and `$0.04351225`
for 945 Gemma 26B calls. The complete artifact is
`STRONGER_GENERATOR_RETRY_CALIBRATION.json`.

## Consequence

The one authorized retry failed. Therefore there will be no profile-prompt tuning,
third attempt, policy run, depth run, structural-gap test, or ranking-fidelity test on
iCRAFT-MD under this project. The external non-myopic empirical claim remains
unestablished. The boundary finding is now more precise: a stronger model can author a
plausible fixed patient support, but the static FactSelect channel still supplies too
few answerable interactions to validate a sequential likelihood model.

Total iCRAFT gate spend, including operationally interrupted attempts and the stronger
retry, was `$0.49412867`. The OpenRouter ledger moved from `$16.81168851` to
`$17.30581718`.
