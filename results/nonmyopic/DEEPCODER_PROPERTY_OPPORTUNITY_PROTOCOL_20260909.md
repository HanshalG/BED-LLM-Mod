# One-bit property experiments: prospective numerical opportunity screen

A different experiment contract, not a retrospective rescue of full-output BED.
An experiment chooses a bounded input and one named public Boolean output test.
It returns only that bit, analogous to a pass/fail test harness; no raw output
is visible to the policy. Each experiment has unit cost. Terminal objective is
prediction of full outputs on separate inputs, using half-multiclass Brier risk.
This observation restriction is an explicit modeling assumption, not a claim
that full outputs are unavailable in every program-synthesis application.

Unchanged source grammar and prior. Four independent empirical-prior panels,
128 program draws, seeds29100000+1000*k+i;40 inputs30100000+1000*k+j.
Eight available actions pair the first8inputs with the eight properties in
their fixed registry order. Last32 inputs are disjoint prediction targets.
Budget4, ordinary receding h1/h2/h3, random, receding openloop h3 and full B4
optimal using the existing exact finite-reference planner. No action/seed search,
no changed budget after results. Existing runtime caps retained, incompletes
reported without rerun. All panels included. Prespecified numerical criterion:
at least5% successive aggregate risk reduction h1->h2 and h2->h3. Report all
values regardless; criterion alone cannot authorize a paper or LLM efficacy.

Properties are total: ERROR true only on failed execution; other tests false
on ERROR. List/empty, positive scalar, list length>=3, negative list member,
nonempty all-even list, nonempty palindrome list; integer values never coerced
to lists. Strict tests precede measurement.

Finite-panel success would still require independent source support/coverage,
truth-anchored predictive validation, LLM hypothesis-role tests, and paired
compute-matched myopic/random/openloop controls on sealed independent cases.
No paid call is part of this screen. Freeze before numerical results.
