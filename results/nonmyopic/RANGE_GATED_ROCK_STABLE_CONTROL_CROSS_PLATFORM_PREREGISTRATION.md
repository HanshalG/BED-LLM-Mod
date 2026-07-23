# Range-Gated Rock Stable-Control Cross-Platform Preregistration

Status: frozen after the local qualification output and before running the new
selector on the cluster.

## Motivation

The seed-24186 proposal producer passed every scientific gate, and independent
replay reproduced every value and aggregate. Its exact-identity audit failed only
because raw floating-point maxima exchanged two tied control identities at roughly
`1e-17` precision across platforms. No trajectory was authorized.

This is a new engineering qualification, not a reinterpretation of that failed
audit. The banked producer and audit remain unchanged.

## Frozen Rule

- A value tie is any value within `1e-12` of the numerical optimum.
- Among tied entries, select the first entry in the already-declared canonical
  order: fixed proposal slot order, legal-action order, or exhaustive-plan order.
- Values are rounded to 12 decimal digits only when constructing the
  cross-platform digest; unrounded values remain the scoring inputs.
- Every non-tied alternative must be more than 100 tolerances below the optimum.
- Every selected identity must remain unchanged under two deterministic,
  alternating perturbations bounded by `1e-12 / 16`.

## Fresh Cluster Gate

The local run over all 16 banked cells passed every mechanic and emitted canonical
digest:

`856105c321d99a4ea3da07d9a2b413b40a7331b55cc6edd952df10a218bf2f8b`

Run the same script and same immutable proposal JSON in the cluster environment on
`msc,llm`, excluding `oat12`, with no model calls. The cross-platform gate passes
only if:

1. every cluster mechanic is true;
2. the cluster canonical digest exactly equals the frozen local digest; and
3. the cluster artifact contains all 16 cells.

A pass authorizes implementation and preregistration of a fresh paired trajectory
protocol using this stable selector. It does not authorize a trajectory by itself.
