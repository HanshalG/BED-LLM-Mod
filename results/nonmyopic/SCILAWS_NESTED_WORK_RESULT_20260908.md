# One-root work trace: repeated inner solves and policy switches

Frozen02ee01fb instrumentation, tested before the one root0/empty-history/h2
diagnostic. Two tests pass in.74s; scoped E4/E7/E9/F lint passes. Instrumentation
matches the uninstrumented terminal value and evaluation count and never invents
a missing second-action score.

Artifact SCILAWS_NESTED_WORK_AUDIT_20260908.json SHA256:
ba31f479f9a64c9ff6717c792c968b18ba05916d82d546e630c3cb7fe6396011.

The same100000 limit terminates this root before completion:
-281 outer callbacks,561 terminal integrations attempted,560 complete.
-99720 terminal integrand evaluations of100001 total (about99.7%).
-Completed terminal solves have median150 and maximum270 evaluations.
-Sampled continuation-action changes bracketed by observations
  [-4.02982532668674,-3.7976756045118356] and
  [.2795985026618574,.2944765065527122].
-The sampled domain spans about-2049.75 to2050.55. High-cost examples include
  both large positive and negative observations; there is no single catastrophic
  inner solve explaining the cap.

These are sampled switches, not a proof there are exactly two or that outer
kinks account for all refinement. The trace does establish that repeated
ordinary-cost inner solves dominate the count; another density micro-optimization
cannot reduce this count. No complete root risk or two-root plan is qualified.

## Evidence-supported next candidate

Prospectively test adaptive integration partitioned at numerically detected
continuation-action switches. Use a deterministic public predictive-coordinate
scan to bracket sign changes, a bounded root solver, and integrate the complete
real line across resulting segments with apportioned absolute tolerance. Every
scan/root-solve inner evaluation must count against the same plan budget.
Do not hardcode these observed switch locations, drop tails, omit an action,
relax error thresholds or regard absent sampled sign changes as proof of none.
Within every segment continue evaluating the actual minimum, so an undiscovered
switch remains visible to adaptive integration. Preserve the unpartitioned path
and verify equivalence on tractable analytic/synthetic integrals before a new
frozen diagnostic. It may still fail the cap; that is not a reason to change it.

No source measurements/model calls, $0. Authenticated account and Sept8 London
ledger unchanged. Process exited, automation paused. Numerical debugging is
not the LLM-native BED result; the full scientific goal remains unfinished.
