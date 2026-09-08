# Full correction panel: no count qualifies

Frozen implementation/protocol bd2d6c37. Result SHA256
34f3835cec1b0f7bd3846d056411cd2697b59c9ceb8916000eab926fe6792516.
Artifact SCILAWS_PARTICLE_CORRECTION_PANEL_20260908/result.json.
All15 shard hashes verified; process exited normally. Three exact banked cases
reused and45 new cases evaluated. All384 independent references reused, not rerun.

| Branches | Cases passing | Worst absolute root error | Worst reference action regret | Seconds/menu |
| --- | --- | --- | --- | --- |
| 4 | 46/48 | 0.000282690 | 0 | 0.0145-0.0198 |
| 8 | 46/48 | 0.000281447 | 0 | 0.0232-0.0375 |
| 16 | 46/48 | 0.000281671 | 0 | 0.0401-0.0525 |
| 32 | 46/48 | 0.000278049 | 0 | 0.0749-0.0890 |

All192 plans completed. No resource failures. The same two cases fail every
count: task2 (lake thermocline), seed1305, affine history; task6 (volcanic column),
seed1304, affine history. Task indices are zero-based. Their32-node worst errors
are0.000278049 and0.000202076, respectively. Thus the promising three-case screen
did not generalize to full-panel qualification. Correct selected actions do not
rescue the independent absolute root-value gate, which matters when values are
used inside future Bellman comparisons.

## Mechanism And Next Constraint

The analytic identity is correct under the finite mixture; this does not make
the residual easy to integrate. For nonzero regression slope, the linear
predictor is unbounded as |Y| grows while the finite-support posterior target
mean is bounded. Consequently the residual squared grows quadratically in the
tails. Quantile quadrature can miss contributions that analytic linear risk
already subtracts. This is a plausible explanation of the nearly flat errors,
not a verified attribution for these two mixtures. Quadrature-node agreement
alone would not establish tail coverage either.

Do not try more counts or discard these cases. A next numerical candidate must
explicitly account for residual tails (for example analytic Gaussian tail
moments with a bounded residual or a bounded predictor with an analytically
integrated risk), and independently test that identity. It must be prospectively
qualified on the same full panel and then continuation histories. This is not
permission to alter observations, likelihoods, targets or count limits. The
banked64-node uncorrected one-step result remains valid on its own terms, but
its deep resource barriers remain unresolved. No low-order deep run is justified.

Seven focused tests1.07s and scoped lint pass. The test assertions verify coverage,
artifact binding, independent assessment and small-model correction identity;
they do not override the scientific null. Zero source measurements, inference
calls and cost. Authenticated account245/220.376693994/24.623306006; London Sept8
spend0. Automation remains paused; full non-myopic LLM goal unfinished.
