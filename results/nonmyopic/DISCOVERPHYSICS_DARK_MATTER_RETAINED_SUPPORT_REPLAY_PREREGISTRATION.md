# DiscoverPhysics Dark-Matter Retained-Support Replay

## Status

Frozen before computing any retained-support posterior or endpoint. This is
a zero-call, post hoc mechanism diagnostic on the exact failed grounded
policy artifacts. It cannot rescue the failed policy or establish fresh
efficacy.

## Source Binding

- Policy SHA-256:
  `ab2b8a4ce3134fd12236e316b126cf00931f15cb78f818d00cca0b3a109abb46`
- Frozen model-state SHA-256:
  `7b13e0a3d105b3c823efe3f7bfe66dfb32f983058c9e0a279cd8f366c5a4b8e0`
- Model responses, initial and refreshed supports, roots, branches,
  continuations, official simulator, hidden-map seeds, noise seeds, Monte
  Carlo counts, held-out experiments, and controls: unchanged.
- New model/API calls and OpenRouter cost: `0/$0`.
- OatML use: none.

## Diagnosis Under Test

Full-history likelihood correction recovered `56.6%` of the original
dynamic-to-fixed gap but remained `9.5%` worse than fixed support. The
residual loss was concentrated in NW and SW, while refreshed support was
better in NE and SE. Replacing the original support after every branch may
therefore discard useful hypotheses even when regeneration adds useful new
ones.

## Frozen Retention Rule

At each frozen branch representative, form a 16-particle mixture:

- mass `0.5` on the exact branch posterior over the eight original
  hypotheses; and
- mass `0.5` on the frozen LLM weights over the eight regenerated
  hypotheses.

Do not deduplicate, tune, or estimate the mixture weight. Treat particles as
model components even when semantically similar.

For every component \(h\), branch representative \(c\), exact root
observation \(y_1\), and continuation observation \(y_2\), update

\[
  p(h \mid y_1,y_2)
  \propto
  m_h w_h
  \frac{p(y_1\mid h)}{p(c\mid h)}
  p(y_2\mid h),
\]

where \(m_h\) is the frozen component mass and \(w_h\) is its normalized
within-component branch weight. All likelihoods use the official simulator.

At \(y_1=c\) with an uninformative continuation, the original and refreshed
components must retain total posterior masses `.5/.5`.

## Replay

Recompute:

- internal trajectory risks and root selection for all four roots;
- fresh-96-map retained-support MSE for center B, myopic D, and random A;
- the unchanged exact two-observation fixed-support center control;
- retained-union nearest-support coverage; and
- the same stratified-bootstrap paired interval.

Common random numbers and all original seeds are reused.

## Frozen Diagnostic Gates

All must pass:

- myopic D and retained-support lookahead B;
- at least `10%` internal risk reduction;
- at least `10%` hidden-map MSE reduction versus D;
- positive paired-bootstrap 95% lower bound;
- at least `5%` gain versus random A;
- at least `5%` gain versus fixed-support B; and
- at least `5%` retained-union nearest-support risk reduction.

A pass is only post hoc evidence that branch-conditioned support should be
retained rather than replaced. It authorizes no automatic rerun. Any fresh
confirmation requires a separately frozen protocol and independent model
outputs.
