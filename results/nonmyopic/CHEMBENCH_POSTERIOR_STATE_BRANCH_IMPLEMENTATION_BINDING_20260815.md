# ChemBench Posterior-State Branch Implementation Binding

Date bound: 2026-08-15 (Europe/London)

- Runner: `scripts/chembench_posterior_state_branch_fidelity.py`
  SHA-256 `05330040923fef91c05d2d241a8a07725bdba69bcf916dddaa929cec181003bf`
- Tests: `tests/test_chembench_posterior_state_branch_fidelity.py`
  SHA-256 `7307cac62bb1a60936f236e64849a8f42266412099f0982f61a8995c6adcf60f`
- Protocol SHA-256
  `ac70e57c8dce66fb7911756ec6d3020e5eff97f4b07e717ec5e0ab19bdab145c`
- Raw-quantile predecessor result SHA-256
  `e4533d76af6ab9be344bbf72806762eb695a425e7664a272105b3d075f706b16`
- Authorized V3 result SHA-256
  `dce0832190aa8b76c9b345220a9043edb6e622db3b8cf0f524b68b5c06c783c1`

Focused tests pass 19/19. One source/easy implementation benchmark completed
in 5.54 seconds with posterior-state Spearman 0.9868, zero pooled top-action
regret, and zero component-bank regret. This runtime-only check did not alter
the frozen panel, seeds, features, clustering, reference count, or gates. The
complete 36-case result has not been computed at this binding.

The superseded implementation hash `bbdb8500...` failed closed before case
7/36 and wrote no result. The protocol-consistent zero-mass-center correction
is documented in
`CHEMBENCH_POSTERIOR_STATE_BRANCH_ZERO_MASS_CORRECTION_20260815.md`.

No LLM, API, network call, benchmark endpoint, or paid resource was used.
