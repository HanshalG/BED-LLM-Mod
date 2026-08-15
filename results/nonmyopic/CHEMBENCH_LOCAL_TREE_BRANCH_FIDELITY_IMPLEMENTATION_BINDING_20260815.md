# ChemBench Local-Tree Branch-Fidelity Implementation Binding

Date bound: 2026-08-15 (Europe/London)

The frozen protocol is implemented by:

- `scripts/chembench_local_tree_branch_fidelity.py`
  SHA-256 `b797f73b91251dcad929582602b96672be63a9af4a27b2d5a0809fb52b63cca4`
- `tests/test_chembench_local_tree_branch_fidelity.py`
  SHA-256 `1c57a066e39388cf7c6b8595845bd2a099aba51022352838988bb36bad7cd8f9`
- protocol SHA-256
  `3bc5d2db761c0e7bb28dab397f155042f8ac5ce97511b2b7533a0ae79f782f8a`

The runner also binds the authorized V3 result SHA-256
`dce0832190aa8b76c9b345220a9043edb6e622db3b8cf0f524b68b5c06c783c1`.
It rejects a changed V3 result or a V3 source-only gate failure before fitting
any branch-fidelity posterior.

Focused tests pass 13/13. A single real source/easy/bank-one implementation
benchmark completed in 0.55 seconds and was used only to check runtime and
finite execution. It did not change the frozen panel, branch counts, seeds,
reference sample count, or thresholds. The complete 72-bank result has not
been computed at this binding.

No LLM, API, network call, benchmark endpoint, or paid resource was used.
