# DiscoverPhysics Dark-Matter Full-History Replay

## Status

Frozen before computing any corrected posterior or replayed endpoint. This is
a zero-call, post hoc mechanism diagnostic on the exact failed grounded
policy artifacts. It cannot rescue the failed policy or support a fresh
efficacy claim.

## Source Binding

- Policy SHA-256:
  `ab2b8a4ce3134fd12236e316b126cf00931f15cb78f818d00cca0b3a109abb46`
- Frozen model-state SHA-256:
  `7b13e0a3d105b3c823efe3f7bfe66dfb32f983058c9e0a279cd8f366c5a4b8e0`
- Model responses, supports, roots, branches, continuations, hidden-map seeds,
  noise seeds, Monte Carlo counts, held-out experiments, and controls:
  unchanged.
- New model/API calls and OpenRouter cost: `0/$0`.
- OatML use: none.

## Diagnosis Under Test

The failed dynamic policy compresses the exact root observation into one of
two branches. Its refreshed-support weights encode the representative branch
observation, and only the continuation likelihood is applied afterward.
The same-center fixed-support control instead retains both exact
observations. This may explain why dynamic center beats myopic by `19.3%`
yet loses to fixed support by `21.9%`.

## Frozen Correction

For refreshed hypothesis \(h\), branch representative \(c\), actual root
observation \(y_1\), continuation observation \(y_2\), and branch-conditioned
LLM weight \(w_h\), compute:

\[
  p(h \mid y_1,y_2)
  \propto
  w_h
  \frac{p(y_1\mid h)}{p(c\mid h)}
  p(y_2\mid h).
\]

All likelihoods use the official simulator and the frozen Gaussian noise.
The ratio preserves the branch-conditioned weight when \(y_1=c\), while
retaining continuous within-branch root information without counting the
representative twice.

No support, weight, branch, action, endpoint, or threshold changes.

## Replay

Recompute:

- internal trajectory risks and root selection;
- fresh-96-map dynamic MSE for center B, myopic D, and random A;
- the unchanged exact two-observation fixed-support center control;
- the same stratified-bootstrap paired interval.

Common random numbers and all original seeds are reused.

## Frozen Diagnostic Gates

The correction is mechanism-consistent only if all original scientific
conditions except the unchanged coverage gate pass:

- myopic D and corrected lookahead B;
- at least `10%` internal risk reduction;
- at least `10%` hidden-map MSE reduction versus D;
- positive paired-bootstrap 95% lower bound;
- at least `5%` gain versus random A; and
- at least `5%` gain versus fixed-support B.

Coverage is not recomputed by the likelihood correction and remains the
original passed `14.4%`.

A pass is post hoc evidence that loss of within-branch root information
caused the fixed-support failure. It authorizes no automatic rerun. Any fresh
confirmation must use a separately frozen protocol and new outputs/endpoints.
