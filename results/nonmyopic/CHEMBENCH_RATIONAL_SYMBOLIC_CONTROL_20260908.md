# Runnable rational symbolic-proposal control

The previous goal turn was progress: the GRN source audit changed the adoption
decision. That interpretation remains closed. This turn fills the missing
symbolic-search input of the new proposal comparison, without opening a new
scientific endpoint or requesting an LLM response.

## Implementation

`environments/chembench_mopen/symbolic_proposer.py` uses the existing
[gplearn genetic-programming implementation](https://gplearn.readthedocs.io/en/stable/reference.html)
rather than a new search algorithm. Optional dependencies are pinned separately
to gplearn 0.4.3 and scikit-learn 1.8.0. An initial 1.7.2 pin was rejected by the
dependency resolver before any fit; the declared dependency requires 1.8.0.

The proposer accepts only real-history input values and noisy log1p-rate
observations, plus a public input box and shared parameter bounds. It accepts no
held-out outcomes, source family identifier, source registry or graph metadata.

Search uses addition, subtraction, multiplication and division, minimizing
training log1p-rate MSE with a size penalty. Negative/nonfinite training rates
receive an explicit large loss. Defaults are population64, generations3 and up
to4 exported laws, with one CPU worker. Hard argument limits are population512,
generations10, 128 history rows, and16 exported laws. At least two initial real
observations are required; an empty-history invocation is not filled with fake
data. Wall time is measured, not forcibly interrupted by this library function;
a future experimental runner must apply its own subprocess deadline.

## Exact semantics and uncertainty

gplearn's division is protected near zero. The downstream IR uses ordinary
division. Exports therefore require an outward-rounded interval check that the
denominator stays outside the protected region throughout the declared input
AND parameter box. There is no silent translation of protected division to a
different function. Whole-box rate nonnegativity is also required; this is a
conservative sufficient check, not a complete validity solver.

The exporter reads actual program trees, never the library's rounded display
strings. Fitted ephemeral constants become separately uncertain parameters with
the shared prior bounds (default independent log-uniform [.01,10]). Their fitted
values are not treated as posterior point masses or used to choose narrow bounds.
Constant-free programs receive a positive amplitude prior including one. Full
history conditioning then occurs in the existing executable belief adapter.
These are parameterized versions of fitted structures: the downstream integrated
forecast is not claimed to equal the native GP fitted-point prediction.

Integration testing caught a fitted positive difference that became negative
after broad parameter draws. The whole-box positivity requirement addresses
this without dropping particles from the posterior. The raw final population,
full-precision constants, native training scores, accepted canonical keys and
rejection reasons are retained, so this restriction is visible and auditable.
If nothing exports, the returned proposal set is empty; downstream complete-arm
requirements must fail the case, not remove it from the endpoint cohort.

## Verification

85 focused tests pass in 4.19 seconds, including the new control, executable
inference, sealed held-out scorer, old IR, expected-policy/context mechanics and
source-audit tests. The search tests use the actual pinned library, not a mocked
optimizer. They verify deterministic repeatability, changed proposals when
observations change, connection to uncertain inference, exact rational-tree
export at original parameter values, protected-division rejection, and declared
resource/input failures before fitting. An isolated test fixture supplies the
Tensor type missing from the repository's torch stub for SciPy compatibility;
production inference is not patched around that test-only issue.

## Scientific limitations

This is a runnable *rational GP control*, not a strongest-possible symbolic
baseline, proof of productive compute matching, or LLM necessity. Its operator
set lacks exponentials and arbitrary real powers available in the LLM IR.
Interval dependency can reject valid expressions; independently uncertain
constants can weaken a good native fitted program. Both restrictions must be
reported. Native GP predictions and stronger compatible search require their
own comparison rather than being silently handicapped to match this adapter.

Search selects candidates by its native fitted-point fitness; the downstream
finite-prior posterior is a separate computation. Better training scores do not
prove better held-out prediction. The work count derived from generation-average
tree lengths is labelled approximate training scalar-node work; it is not total
wall-clock effort, tokens, posterior integration work, or a claimed equality of
compute across methods.

No new source-grounded planning opportunity has passed, and no LLM semantic gate
or paired policy study is authorized by these mechanics tests. The full plan
still requires that opportunity, a complete prospective LLM interface and fair
controls, predictive calibration, and fresh paired policy/confirmation evidence.
Cost $0; no cluster, cleanup, closed-endpoint rerun or automation change.
