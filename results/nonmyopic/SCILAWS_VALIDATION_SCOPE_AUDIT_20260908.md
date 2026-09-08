# Scope audit: initialized instrument versus the research goal

Read GOAL.md, the third-pass plan, reference prior/measurement protocols,
reference_prior.initialize, diagnostic constructors and newest banked results.
The preceding numerical changes are progress, not completion of the plan.

## Concrete integration change

Added initialize_corrected as an explicit path; old initialize remains unchanged.
It applies the original shared response scaling and all initial updates, then
constructs the corrected model with those posterior components and weights.
Its model.initial_state is the conditioned posterior, preventing an accidental
zero-data reset when callers use that property. It does not run or score a policy.

Frozen963fbac4 contract audit80db35cd7df2b3379b568dc8864780e6fcadbc1f7151d38a301fd4b54fb790ce
passes24/24 cases: eight fixed designs x zero/affine/quadratic synthetic initial
tables, no source labels. Maximum posterior-risk difference is0.12 focused
initialization/prior tests pass in1.31s; scoped E4/E7/E9/F lint passes. Tests
also compare subsequent conditioning and full8-action/64-target coverage.
These deliberately simple synthetic tables are software fixtures, not evidence
of scientific realism, calibration, model coverage or planning opportunity.

## Requirement coverage

| Requirement | Current evidence | Verdict |
| --- | --- | --- |
| Shared6/10/14 initial observations, frozen scaling | initialize +24 contracts | Wired and tested |
| Full8-action/64-target/4-family posterior state |24 contracts | Wired and tested |
| Corrected initialization without prior reset | New explicit adapter +8 update comparisons | Tested |
| Full-geometry posterior-conditioned numerical accuracy | Recent accuracy panels use2 actions and2 scalar families | Missing |
| Complete h2 stress-panel runtime/accuracy | Quintic3/4 histories, empty cap | Incomplete |
| Generic exact contingent h3 semantics | Earlier small exact planner tests | Small-case mechanics only |
| Deployable accelerated h3 on full geometry | Value surrogate is h2-only | Not implemented/qualified |
| Receding four-measurement source episode | No new source observations | Missing |
| Independent source-model calibration | Working Gaussian residuals and empirical-Bayes scaling | Unverified |
| Useful LLM executable proposals and real-history advantage | No fresh SciLaws proposer responses | Missing |
| Paired h1/h2/h3, open-loop, compute-matched and random controls | No source pilot or sealed final seeds/replicates | Missing |
| Source usage/attribution resolution | Prior audit leaves obligations open | Incomplete |
| Anticipating future LLM discovery | Current numerical support fixed within search | Missing |

The zero-history stress failure is not erased because the intended policy is
initialized. Conversely, a stress failure on a2-action prior is not evidence that
the initialized8-action source experiment has no opportunity. Neither inference
is warranted. Keep the tests, their roles and their permissions separate.

## Next concrete work

Freeze an initialization-conditioned h1 numerical comparison on all eight designs
and the three declared software fixtures, with independent adaptive reference and
the full action menu. This establishes whether the new adapter and algorithms
actually agree on the state shape the intended experiment will use, without
opening private labels or claiming source efficacy. Preserve all current caps
and failed stress results. Report incomplete references explicitly. Only after
that evidence should a full-geometry h2/h3 implementation decision be made.

Do not claim that this alone opens a source or paid stage. The table above is the
remaining objective, not an invitation to redefine completion around numerical
fixtures. No source/model/planning calls in this contract audit, $0; account and
Sept8 London ledger unchanged. Process exited, automation paused, goal unfinished.
