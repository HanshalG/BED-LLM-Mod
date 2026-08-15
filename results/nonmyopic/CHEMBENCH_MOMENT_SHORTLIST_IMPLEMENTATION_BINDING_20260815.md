# ChemBench Moment-Shortlist Implementation Binding

Date bound: 2026-08-15 (Europe/London)

- Runner: `scripts/chembench_moment_shortlist_sampling.py`
  SHA-256 `4c317c4d6083caf41609be8a3686e74001b3d92fa495b49f65bcbc48456247c1`
- Tests: `tests/test_chembench_moment_shortlist_sampling.py`
  SHA-256 `f29e88f7093e501668426a9070e5dfb92547d4e608b55eeae17f7ad8ad657a4b`
- Protocol SHA-256
  `ac30b9c79527976cf86939232b317d9d3769dc3fd7608370cd760674d6167266`
- RQMC predecessor SHA-256
  `a966d5cf4984c9907649a0dae5d6bb8a19982f942c83f463f7ec61e4e2d439f2`
- IID predecessor SHA-256
  `d71964ff247bc80408b0b5c78c4b168c5117ece372446990c4a5dfd17b42c0f6`
- Pooled and component reference SHA-256
  `637c031946268e8015687d5eb4af4d037bfc1399300e56c0e50bdf049b4fab07`
  and `e4533d76af6ab9be344bbf72806762eb695a425e7664a272105b3d075f706b16`.

Focused tests pass 27/27. They bind moment-proxy ordering, stable assay-index
tie breaking, and full-panel regret when a shortlist excludes the full best
action. The runner reconstructs every pooled posterior and uses shared IID
particle/noise streams across the endpoint-blind top-eight actions. The full
36-case result has not been computed at this binding.

No LLM, API, network call, benchmark endpoint, or paid resource was used.
