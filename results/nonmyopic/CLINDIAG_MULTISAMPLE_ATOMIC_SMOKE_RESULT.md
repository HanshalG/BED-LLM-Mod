# ClinDiag Multisample Atomic Serving Result

Date: 2026-07-24

Run: `clindiag-multisample-atomic-smoke-20260724T223037Z`

Status: **passed every frozen serving gate; the four-case opportunity gate is
authorized unchanged.**

The run completed exactly 26 physical requests: 24 full GPT-5.4 support
generations and two GPT-5.4 Mini joint semantic measurements. All 24 supports
parsed to exactly 12 distinct diagnoses. Reasoning, structured retries, forced
exits, and runtime failures were zero.

Cost was `$0.05909625`. Linear request-count projection for the 608-call
opportunity gate is `$1.38194`, below the preregistered `$3.00` projection and
`$8.00` hard cap.

Private raw responses were checkpointed under SHA-256
`2df0513de35a25500159bc49027c3cf3d175a383cb8e511c70ea38c01a520b64`.
No smoke scientific endpoint or threshold is used by the formal gate.
