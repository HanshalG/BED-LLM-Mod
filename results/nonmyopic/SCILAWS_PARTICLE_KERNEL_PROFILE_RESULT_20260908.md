# Kernel profile: normalization helps, but weak bounds still force work

Frozen profile0f1de83e. Artifact SHA256
68b29aeca6dbd03378152f0acbe511c8023af6837d786802cf4705be12ab6d44:
SCILAWS_PARTICLE_KERNEL_PROFILE_20260908.json.
Three fixed first-task fixtures,2056 point evaluations per timing repetition,
three alternating repetitions per implementation. No policy/reference integral
rerun or new outcomes. Process exited normally.

| History | Median old seconds | Median max-shift seconds | Microkernel speed ratio |
| --- | --- | --- | --- |
| zero | 0.3046 | 0.1478 | 2.06 |
| affine | 0.2803 | 0.1781 | 1.57 |
| quadratic | 0.2418 | 0.1204 | 2.01 |

Maximum pointwise density-times-risk discrepancy<=1.95e-16. cProfile identifies
NumPy reductions as the largest own-time category and centered risk as the next
major cost. The reduction category also includes sums/checks, so its entire time
must not be attributed only to logaddexp. Separate alternating measurements
establish the actual normalization effect. Timings vary and are not confidence
intervals or end-to-end speedups. Production reference implementation unchanged.

## Why Not Rerun The Same Planner Yet

The last workload did not finish even one root out of8 in5s. A roughly2x local
speedup does not establish full-decision feasibility; integration count can also
become binding once CPU is faster. More importantly, the current action lower
bounds are all the SAME expected irreducible target noise at a given child.
Whenever the incumbent upper bound exceeds that noise, no unrefined action can
be discarded by these bounds. The scheduler consequently integrates essentially
every action in an important child. This is a structural pruning limitation,
not merely a Python overhead issue.

Next prioritize a genuinely action-dependent lower bound, independently derived
and tested. One candidate follows directly from the finite-mixture risk identity:

    expected mean-risk = sum_{i<j} w_i w_j ||f_i-f_j||^2
                                      integral l_i(y) l_j(y)/p(y) dy.

Cauchy-Schwarz and integral p=1 give

    integral l_i l_j/p >= (integral sqrt(l_i l_j))^2.

For scalar Gaussian likelihoods the overlap squared is

    2 sigma_i sigma_j/(sigma_i^2+sigma_j^2)
      * exp(-(mu_i-mu_j)^2/[2(sigma_i^2+sigma_j^2)]).

Thus a pairwise overlap sum plus expected target noise is a closed-form,
action-dependent lower bound. This is a derived candidate, not a novelty claim
or evidence that it will be tight enough. It recovers current risk for identical
likelihoods and can distinguish informative actions, unlike the noise-only bound.
Its O(P^2) cost/memory must be measured and bounded with streaming, especially
at2048 particles. Do not implement a dense unbounded intermediate or assume the
bound pays for itself. Unit-test against independent integrals before a frozen
bound-tightness/cost audit, then decide whether full search is worth testing.

The max-shift prototype is retained for later independent integral equivalence
tests, not silently substituted into banked references. No new refinement grid
launched. Three tests passed0.88s; lint passed; all earlier scientific gates and
nulls unchanged. Source/model calls0, paid cost0, account/ledger unchanged,
automation paused, full LLM-native non-myopic research goal unfinished.
