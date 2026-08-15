# ChemBench RQMC Sampling Implementation Binding

Date bound: 2026-08-15 (Europe/London)

- Runner: `scripts/chembench_rqmc_sampling_fidelity.py`
  SHA-256 `8261f6e9e212665fb3e905e45c700f6fdbd70008bff7061b332b50312e70e15a`
- Tests: `tests/test_chembench_rqmc_sampling_fidelity.py`
  SHA-256 `001dff7f49242bbee0f130ce89b147232795bdfbcb8e580665932b4d82526c82`
- Protocol SHA-256
  `312cbc70d4f34df777e0cc5f35afc4c7779eb9073341b187ab4c8c6f8d939791`
- IID predecessor SHA-256
  `d71964ff247bc80408b0b5c78c4b168c5117ece372446990c4a5dfd17b42c0f6`
- Pooled reference SHA-256
  `637c031946268e8015687d5eb4af4d037bfc1399300e56c0e50bdf049b4fab07`
- Component reference SHA-256
  `e4533d76af6ab9be344bbf72806762eb695a425e7664a272105b3d075f706b16`

Focused tests pass 24/24. The runner reconstructs and verifies each pooled
posterior, generates four independently shifted nested Sobol prefixes, shares
each prefix's particle and normal coordinates across actions, and delegates to
the unchanged IID rank/regret gate implementation. The complete 36-case result
has not been computed at this binding.

No LLM, API, network call, benchmark endpoint, or paid resource was used.
