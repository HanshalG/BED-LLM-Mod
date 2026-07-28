# DiscoverPhysics Default-Routing Structured Replication V3 Result

## Status

`REPLICATION_NULL`

The fresh structured support tree passed every Phase-A gate and the
non-myopic B-versus-D result replicated on 384 fresh maps. The LLM-native
regenerated-support contribution did not replicate: retained B was
significantly worse than same-root fixed-support B.

No rerun, alternate mixture, subset, schema, or threshold is authorized.

## Frozen Run

- Protocol/code commit: `e9d18a5`
- Interface:
  `discoverphysics-dark-matter-structured-replication-2`
- Run:
  `discoverphysics-dark-matter-structured-replication-v3-20260728T063000Z`
- Official DiscoverPhysics commit:
  `33b7fa9df96de9c35744efd181ca7e5a8dd60ad5`
- Model-frozen SHA256:
  `473cf5c883929a2cf8b6d862bebf69e1bb6b6401bea8c7b0edba955e847b7e1d`
- Policy SHA256:
  `917e161308d06b648aee9049e7e47a1c608627db7fcd9aafe6930dee2fa2e2ff`
- Private raw-response SHA256:
  `4863a49024c3be1cd0359220e2d9289334db7d62d1bac0b0d232cf1307b0e5ee`

## Serving And Phase A

Default provider routing fixed V2's pre-routing `404`. The strict schemas
returned ten exact compilable responses with no retry or repair.

| Quantity | Value |
|---|---:|
| Accepted responses / HTTP attempts | `10 / 10` |
| Discarded preflight / scientific calls | `1 / 9` |
| Prompt / completion tokens | `15,242 / 6,248` |
| Reasoning tokens | `0` |
| Retries | `0` |
| Forced exits/finals | `0 / 0` |
| Cost | `$0.131825` |

All mechanics gates passed:

- all eight refreshes changed support;
- all four roots had branch-distinct supports;
- center B chose `r4.5_a5` and `r4.5_a2` in its two branches;
- every refresh retained at least two regions; and
- request, transport, reasoning, forced-exit, and budget accounting were exact.

The exact internal model selected the preregistered roots:

| Root | Immediate EIG | Retained internal trajectory risk |
|---|---:|---:|
| A | `1.1164` | `1.5089` |
| B | `1.2767` | `0.4955` |
| C | `0.9391` | `0.8709` |
| D | `1.4058` | `1.0787` |

Immediate EIG selected D; retained lookahead selected B. The predicted B-versus-D
internal risk reduction was `54.06%`, far above the frozen `10%` gate.

## Fresh Physical Endpoint

The frozen policy then opened exactly seeds `24700--24717`: 384 maps, 96 per
region, common observation noise, and 10,000 stratified bootstrap samples.

| Policy | Weighted trajectory MSE |
|---|---:|
| Retained non-myopic B | `3.2446` |
| Retained myopic D | `3.7284` |
| Retained random A | `6.5050` |
| Same-root fixed-support B | `3.0709` |

| Frozen comparison | Result | Gate |
|---|---:|---:|
| B reduction versus D | `12.98%` | pass, at least `10%` |
| Paired CI for `MSE(D)-MSE(B)` | `[0.2917, 0.6758]` | pass |
| B reduction versus random A | `50.12%` | pass, at least `5%` |
| B reduction versus fixed B | `-5.66%` | **fail**, at least `1%` |
| Paired CI for `MSE(fixed B)-MSE(B)` | `[-0.2448, -0.1050]` | **fail** |
| Nearest-support risk reduction | `27.80%` | pass, at least `5%` |

The conjunction therefore fails. The non-myopic root advantage survives a new
LLM tree, but branch-generated support is not robustly value-adding.

## Where The Simulator Diverged

The endpoint is region-heterogeneous:

| Region | B vs D | B vs fixed B | Mean refresh posterior mass |
|---|---:|---:|---:|
| NE | `+33.12%` | `+8.44%` | `10.76%` |
| NW | `-31.51%` | `-33.41%` | `33.48%` |
| SW | `-31.96%` | `-7.74%` | `19.07%` |
| SE | `+67.64%` | `+14.57%` | `71.01%` |

B beats fixed B on 259/384 maps, but its large NW/SW losses dominate many
smaller gains under the frozen regional prior. The B north/east branch's
generated support assigns `95.19%` of its internal mass to NE and only `3.85%`
to NW, while its south/west branch assigns `64%` to SW.

A zero-call posterior decomposition exactly reproduced retained B
(`1.8e-15` maximum absolute error) and fixed B (`9.8e-14`). It found:

- the nominal `5%` refreshed component becomes `25.26%` mean final posterior
  mass;
- refresh-only MSE is `3.4445` versus `3.0709` for the initial component;
- the refreshed component is better on `60.48%` of weighted events, but its
  mean posterior mass is only `22.87%` when better and `28.92%` when worse;
- refresh posterior mass correlates `-0.311` with refresh predictive advantage;
- in NW, refresh-only MSE is `3.2610` versus initial `2.1149`, yet refreshed
  mass rises to `33.48%`; and
- in SE, refresh-only MSE is better (`2.2747` versus `3.0252`) and mass rises
  appropriately to `71.01%`.

This localizes the failure to likelihood-weighted support calibration. Fresh
hypotheses improve nearest-map coverage, but matching two probe observations
does not reliably identify hypotheses with good held-out trajectories. The
posterior can amplify a branch-skewed regenerated component exactly where its
target prediction is worse.

## Comparison To The Original Tree

The original frozen tree achieved:

- B versus D: `22.64%`;
- B versus fixed B: `+2.41%`, paired CI `[0.0341, 0.1100]`; and
- nearest-support gain: `22.15%`.

Even there, retained B's internal risk (`0.4523`) exceeded internal fixed B
(`0.3789`) by `19.4%`. On the new tree the internal retained penalty grows to
`75.8%` (`0.4955` versus `0.2819`) and the fresh endpoint reverses. The sign of
an internal fixed-support comparison is therefore not a valid gate by itself,
but its magnitude is a useful warning about support-tree instability.

## Scientific Conclusion

The previous same-root LLM-support gain was real on its frozen fresh-map
endpoint but is not generation-robust. This replication strengthens the more
limited result that simulator-grounded non-myopic root selection can survive a
fresh LLM support tree. It rejects the stronger claim that this particular
`.95/.05` path-dependent support update reliably improves the selected policy.

The next method should score or calibrate regenerated support against
held-out-predictive consistency, not merely probe-observation likelihood or
nearest-map coverage. Any such method requires a new prospective development
protocol and fresh endpoint; V3 itself is closed.

## Accounting

- New model calls: `10`
- OpenRouter cost: `$0.131825`
- Authenticated remaining balance: `$8.858329844`
- Reserve: none
- OatML, Slurm, SSH, or cluster use: `0`
