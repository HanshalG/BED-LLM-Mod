# ChemBench Posterior-Sampling Implementation Binding

Date bound: 2026-08-15 (Europe/London)

- Runner: `scripts/chembench_posterior_sampling_fidelity.py`
  SHA-256 `cc1ececf50ef0931e8aea0e402c68b6cabb704249d1dc744e978c4a62c7b6a66`
- Tests: `tests/test_chembench_posterior_sampling_fidelity.py`
  SHA-256 `45218eecbb2c776ce553fa5fc0f46933fe2c9e1f3b2e88ee5f7bea52c4d158db`
- Protocol SHA-256
  `b740e2a31b4f5ecc57b07661fb47b4f48bd173ca3aff814be735a1089a3bd542`
- Pooled predecessor SHA-256
  `637c031946268e8015687d5eb4af4d037bfc1399300e56c0e50bdf049b4fab07`
- Component reference SHA-256
  `e4533d76af6ab9be344bbf72806762eb695a425e7664a272105b3d075f706b16`
- Authorized V3 SHA-256
  `dce0832190aa8b76c9b345220a9043edb6e622db3b8cf0f524b68b5c06c783c1`

Focused tests pass 21/21. The runner reconstructs and hashes every pooled root,
regenerates all 2,048 ordered outcomes, and requires each regenerated full
sample mean to match the saved pooled reference before scoring the four nested
replicates. The complete 36-case sampling result has not been computed at this
binding.

No LLM, API, network call, benchmark endpoint, or paid resource was used.
