# Independent adaptive reference: terminal error localized

Frozen code/protocol c174684e, executed once. Two focused tests pass in.80s,
including independent integration of full conditioned posterior risk versus the
analytic-within plus adaptive-between decomposition. Scoped E4/E7/E9/F lint
passes. No production solver changes or source measurements.

Artifact SCILAWS_ADAPTIVE_REFERENCE_AUDIT_20260908.json SHA256:
4096b0febc48056ff79d7089688357fa565d37d4ae6c101c7530de59953c3589.

All four h1 references complete in240-420 integrand evaluations over both roots.
Maximum reported terminal integration error estimate is6.02e-9. Maximum all-root
absolute difference versus the adaptive terminal reference:

| History | Order4 | Order8 | Order16 | Order64 |
| --- | ---: | ---: | ---: | ---: |
| Empty | .01137562 | .00369936 | .00021408 | 2.63e-9 |
| Positive | .00055430 | .00007669 | .00000305 | 5.51e-9 |
| Negative | .00156153 | .00008353 | .00000357 | 5.64e-9 |
| Contradictory | .00125015 | .00001931 | .00000033 | 1.88e-10 |

This independently localizes a failure in the terminal nonlinear between-family
integral: orders8 and16 fail the existing1e-4 scale on the empty-history case,
although the conjugate within-family integral is exact. The analytic correction
and family-bound validity did not remove this residual. Order64 agrees closely
here; this says nothing automatic about deeper adaptive minima or source worlds.

All four h2 nested reference attempts hit the unchanged five-second cap after
47102-54140 total integrand evaluations. No h2 root set or chosen policy was
completed or substituted from earlier results. Encountered inner error estimates
range up to5.20e-6 because QUADPACK also uses relative tolerance at extreme
conditional risks; these are reported, not promoted to rigorous global error
bounds. No h2 reference is qualified by this run.

## Next numerical step

The independent terminal integral is usable as a diagnostic; nested cost remains
the blocker. Profile and precompute invariant Student-t density constants in the
adaptive integrand, validate against scipy.stats.t over central and tail inputs,
and preserve the identical infinite-domain integration, tolerance, action menu
and resource caps. This is an exact algebraic acceleration candidate, not a
smaller rule or threshold change. Check whether it completes the same bounded
opened h2 diagnostic before considering wider source planning. If counted
evaluation limits still prevent completion, do not call the reference solved.

The scientific task and LLM role remain untouched and unproven. No source values
or modelcalls, $0. Account245/220.376693994/24.623306006 and Sept8 London spend0
unchanged; process exited; automation paused; full goal incomplete.
