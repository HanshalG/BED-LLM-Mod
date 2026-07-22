# Mushroom Feature Acquisition Depth Qualification Result

## Decision

The preregistered exact qualification passed. This non-spatial semantic task contains
a reproducible depth-two advantage and is authorized for a separate LLM serving
smoke. No LLM response was used in the qualification.

## Result

The empirical prior is uniform over 8,124 UCI Mushroom catalog rows. Five field
features are initially available; collecting a specimen costs one round, yields no
observation, and permanently unlocks 17 detailed features. Planning minimizes
expected cumulative edible/poisonous posterior entropy over its local horizon.

| Endpoint (d2 - d1) | Mean gain | Paired 95% CI | W/T/L |
| --- | ---: | ---: | ---: |
| Entropy AUC | +0.115348 | [+0.105676, +0.125362] | 693/0/307 |
| Truth-log AUC | +0.117191 | [+0.102234, +0.132444] | 542/0/458 |
| Final entropy | 0.000000 | [0.000000, 0.000000] | 0/1000/0 |

The hidden rows were sampled without replacement at fresh seed `24123` and paired
across policies. All likelihoods and observations were deterministic catalog lookups.

## Mechanism

| Round | d1 entropy | d2 entropy |
| ---: | ---: | ---: |
| 1 | .556368 | .692501 |
| 2 | .369863 | .059965 |
| 3 | .258915 | .017898 |
| 4 | .200257 | .004875 |
| 5 | .183686 | .000000 |
| 6 | .128931 | .000000 |
| 7 | .000000 | .000000 |
| 8 | .000000 | .000000 |

Depth two deliberately accepts zero information in round one by collecting the
specimen, then queries odor in round two on every trial. Depth one starts with
population, and its class posterior does not reach zero entropy until round seven.
Both policies saturate at the endpoint, but their entropy and truth-log AUCs remain
well separated.

An independent trace replay reconditioned the raw catalog posterior after every
stored action and observation, reproduced all metrics and paired comparisons, and
obtained positive independent bootstrap lower bounds for both registered endpoints.

## Scope

This establishes an exact planning opportunity, not an LLM result. The next gate is a
10-cell non-thinking 26B indexed-policy smoke. Its key question is whether semantic
feature choice, especially selecting odor after specimen collection, beats matched
random proposals without exposing EIG scores or hidden rows.

Source: UCI Machine Learning Repository, *Mushroom* dataset,
https://doi.org/10.24432/C5959T (CC BY 4.0).

## Artifacts

- `mushroom_feature_depth_qualification_20260723/REPORT.json`
- `mushroom_feature_depth_qualification_20260723/REPORT.md`
- `mushroom_feature_depth_qualification_audit_20260723/AUDIT.json`
- `mushroom_feature_depth_qualification_audit_20260723/AUDIT.md`
