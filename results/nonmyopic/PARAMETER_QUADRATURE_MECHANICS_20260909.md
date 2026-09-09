# Resolution-checked integration: contract implemented, difficult fits unresolved

Implementation and diagnostic frozen at e0ccdeff. No API calls, cost or scientific
endpoints. Existing executable-particle and SMC modules remain unchanged.

The new parameter_quadrature utility uses SciPy Gauss-Legendre rules with the
correct normalized independent-uniform prior in transformed coordinates. It
returns weighted nodes, moments and conditional evidence only after two consecutive
refinements agree in all requested moments and evidence. It explicitly supports
one or two parameters and at most 16 predictive coordinates, orders up to 512,
and a cumulative 400000 evaluated-row ceiling. Unsupported dimensions and
unresolved integrals are errors, not a fallback to the old 32-particle fit.

Twelve focused tests pass in .83 seconds (including previous reference tests).
New tests cover uniform prior normalization/moments, log-uniform decoding,
Gaussian posterior/evidence against closed-form values, pre-callback caps,
nonfinite/zero evidence and refusal of an unresolved narrow posterior. Gaussian
finite-bound tail corrections are below the stated test tolerance.

## Measured outcome

The one-shot diagnostic on both opened reference fixtures returned **unresolved**:
neither produced two successive agreements before the fixed final order. The
artifact is PARAMETER_QUADRATURE_REFERENCE_20260909.json, bound to backend SHA
f3a33ec43cade6e446e03c597709cde2cdaff606432c49f43f1fd1ebb48e3e99.
No fit, evidence ranking or posterior was accepted. This does not establish that
the finest rule is inaccurate: its intermediate estimates were not retained by
this version, so the result supports only failure of its resolution-agreement
contract. It is an instrumentation limitation, not a new scientific null.

Agreement across finite rules also cannot certify absence of an unseen narrow
mode. In particular, the utility's interpretation explicitly says resolution
agreement, not coverage certification. A passed numerical test would not by itself
justify arbitrary generated structures or correct data-dependent model selection.

## Next action

Do not connect this unresolved backend to the LLM proposer or declare the inference
dependency solved. The next engineering iteration should retain per-refinement
evidence/moment diagnostics and evaluate adaptive integration from an established
numerical library, with independent analytic and adversarial regression fixtures.
These opened numerical fixtures are development tests, not sealed scientific
cohorts: future engineering may improve a backend transparently without changing
any old result or claiming a newly held-out success. Avoid another particle-count
sweep or paid model retry while integration remains unresolved.

The final target remains semantic, path-dependent hypothesis generation plus
ordinary non-myopic sequential BED with productive compute-matched controls and
paired endpoints. This turn changes implementation and yields a negative numerical
readiness result, not a narrower substitute for that objective. Previous turn was
progress; goal active and incomplete. Authenticated balance 23.693468061 and
conservative remaining London-day allowance 4.11174654 are unchanged.
