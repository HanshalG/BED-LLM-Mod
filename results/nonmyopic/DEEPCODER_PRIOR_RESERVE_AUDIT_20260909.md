# Independent prior reserve does not resolve exact-answer support

Retrospective diagnostic on the four opened transition cases; not a rerun or
cap increase of the earlier64-history rejection gate. Exactly4096 source-prior
draws per case, seeds27100000+10000*case+i. Retain multiplicities and only
programs matching both initial observations. No outcome-directed selection,
early stopping at a favorable sample count, or adaptive draw increase.

| Case | Retained /4096 | Samples predicting actual third answer |
|---|---:|---:|
| 0 | 2 | 2 |
| 1 | 1 | 1 |
| 2 | 5 | 3 |
| 3 | 201 | 0 |

Case3 reserve includes103 empty-list predictions and98 others (including17
ERROR outcomes), but no[7]. Thus it can suggest alternative outcomes, yet still
misses the exact answer responsible for the observed model discovery. The
other cases' surviving counts are far too small to infer reliable probability
estimates. Do not interpret probabilities1 from one/two samples as certainty.
These results do not rule out the true full-prior posterior assigning positive
mass, nor prove the LLM can never model surprises. They reject this small
empirical reserve as an adequate exact-answer simulator on this diagnostic.

No predictive mixture coefficient chosen or evaluated after seeing outcomes.
No prior gate rescued, no target-based support insertion, no depth claim.
Conditional accepted draws have the intended prior-rejection interpretation,
but finite samples do not guarantee full predictive support or calibration.

Next architectural candidate: executable property observations rather than
complete output vectors (e.g. emptiness, error, length or element predicates).
This changes the experiment interface and must be independently justified and
frozen, not silently applied to improve the old null. Binary/coarse observations
may reduce exact-output sparsity and allow feasible model-conditioned transition
branches; they do not guarantee a non-myopic gap. Required checks remain source
semantics, posterior/predictive coverage, genuine ordinary horizon headroom,
fresh LLM proposal quality and matched myopic/random/open-loop controls.
This targets semantic program discovery, not an ornamental LLM action selector.

API calls/spend0, no new hidden outcomes. Test fixed draw count, multiplicity,
full-history filtering and validation passed.22s; lint passed. Account unchanged
at usage220.458278479; prior uncertainty retained. Previous goal turn progress;
this turn contributes negative simulator evidence and a precise interface issue.
The requested non-myopic result remains unachieved.
