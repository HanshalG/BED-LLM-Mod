# Bongard OpenWorld Sample-Size Power Audit

Date: 2026-08-08. Model calls: `0`. Endpoint data accessed: `false`.

This is a design-sensitivity calculation, not an outcome forecast. It
uses the exact frozen relative-Brier and evidence thresholds while
leaving ranking, log-loss, and multi-control conjunctions unmodelled.
Actual full-tier power is therefore no greater than any listed marginal
effect-gate power.

## Paired-Effect Sensitivity

`SD ratio` means paired task-difference SD divided by control mean Brier.
The final column is the true relative gain needed for 80% marginal power
on the joint 3% effect-size and evidence gate.

| Design | N | SD ratio | Observed gate | True gain for 80% power |
|---|---:|---:|---:|---:|
| development32 | 32 | 20% | 3.00% | 5.98% |
| development32 | 32 | 30% | 4.46% | 8.93% |
| development64_candidate | 64 | 20% | 3.00% | 5.10% |
| development64_candidate | 64 | 30% | 3.16% | 6.31% |
| confirmation64 | 64 | 20% | 4.90% | 7.00% |
| confirmation64 | 64 | 30% | 7.35% | 10.51% |
| confirmation96_candidate | 96 | 20% | 4.00% | 5.72% |
| confirmation96_candidate | 96 | 30% | 6.00% | 8.58% |

## Changed-Path Sensitivity

The exact calculation requires both the total changed-path threshold
and at least one change in every fixed execution block.

| Design | Per-task change rate | Gate probability |
|---|---:|---:|
| development32 | 37.5% | 55.3% |
| development32 | 50.0% | 93.6% |
| development64_candidate | 37.5% | 54.7% |
| development64_candidate | 50.0% | 98.4% |
| confirmation64 | 37.5% | 54.7% |
| confirmation64 | 50.0% | 98.4% |
| confirmation96_candidate | 37.5% | 53.8% |
| confirmation96_candidate | 50.0% | 99.5% |

## Daily-Cap Feasibility

| Design | Tasks/block | Accepted | HTTP attempts | Exposure |
|---|---:|---:|---:|---:|
| development32 | 8 | 344 | 351 | $1.404 |
| development64_candidate | 16 | 688 | 702 | $2.808 |
| confirmation64 | 16 | 688 | 702 | $2.808 |
| confirmation96_candidate | 24 | 1032 | 1053 | $4.212 |

## Decision Boundary

The frozen 3% value is a minimum claim threshold, not a true-effect
target with 80% power. At moderate 20--30% paired SD ratios, moving
development from 32 to 64 tasks and confirmation from 64 to 96 tasks
materially lowers the detectable true gain while fitting the existing
hard `$5` daily cap. Any expansion must be frozen before planner or
endpoint responses, retain disjoint opaque tasks, scale changed-path
counts proportionally, and leave every semantic and efficacy gate
otherwise unchanged.
