# MQTT: depth gains, but no advantage over fixed experimental design

The input-only target menus and independently tested reference were frozen and
pushed as f997836d before the single complete source-prior execution. All eight
groups completed within the predeclared 180-second per-group cap. No group was
excluded, no retry occurred, and no model call was made. Result SHA256:
f87f542309f320a219e007fd158724b268be1e56a1d0864844c681508972e3a4.

## Results

Expected terminal half-Brier risk on 512 fixed six-command target words per group;
lower is better. All policies have the same six physical-command budget including
resets. h1/h2/h3 are ordinary receding horizons, not renamed policy improvement.

| Source group | h1 | h2 | h3 | Optimal fixed sequence | Random |
|---|---:|---:|---:|---:|---:|
| invalid | 0 | 0 | 0 | 0 | .08309739 |
| mosquitto | 0 | 0 | 0 | 0 | 0 |
| non_clean | .02343750 | 0 | 0 | 0 | .07067296 |
| simple | 0 | 0 | 0 | 0 | .03831138 |
| single_client | .00205078 | .00205078 | 0 | 0 | .01606950 |
| two_client | 0 | 0 | 0 | 0 | .08505620 |
| two_client_same_id | 0 | 0 | 0 | 0 | 0 |
| two_client_will_retain | .00419922 | .00419922 | .00026042 | 0 | .04634559 |
| Equal-group mean | .00371094 | .00078125 | .00003255 | 0 | .04244413 |

Mean h1-to-h2 reduction is exactly 15/19 (78.95%); h2-to-h3 is 23/24
(95.83%). Full adaptive B6 also has zero risk in every group. Therefore the
required adaptive-versus-committed advantage is absent in all eight groups;
its relative denominator is zero, not a passing percentage. The frozen gate
fails. Five groups have no h1-to-h3 difference, one improves only at h2, and
two improve only at h3. These are exact finite-prior expectations, not independent
population estimates or a powered statistical result.

## Independent Verification

Thirteen focused tests pass. An uncached explicit-world/history implementation
agrees with adaptive, receding, random and committed reference values on five
synthetic stateful ensembles. Another exhaustive fixture verifies a real adaptive
advantage over every fixed sequence, showing the control is not inadvertently
adaptive. Reset tests preserve posterior information while resetting physical state.

After execution, a separate direct trajectory implementation replayed each saved
fixed sequence across every source model and recomputed posterior predictive
half-Brier risk without invoking the planning reference. All eight are exactly
zero. Since Brier risk is nonnegative, these feasible sequences independently
certify that adaptive planning cannot improve on this fixed control for this
precise target panel. All saved source/reference/menu bindings and group records
were checked; certificate is banked beside the result.

Some terminal partitions still contain multiple source models. Zero *target*
risk does not imply complete machine identification or equivalence on untested
words. The singleton is trivial; the three-model same-ID group also has zero
initial risk on this target menu. Neither is counted as a scientific success.

## Research Decision

Close this exact source-prior, budget and target panel. Do not decrease the
budget, change target words, remove easy groups or weaken the committed control
to manufacture a passing result. The source contains stateful command-prefix
effects, but published-model classification with only one to five candidates
does not supply the needed adaptive advantage here. A headline based only on
the large relative h1/h2/h3 gains would omit the decisive stronger baseline.

This is not an LLM failure: no LLM participated. It also does not rule out
open-world protocol learning, which would be a different inference problem.
Before any such descendant, require an independently motivated source contract
for discovering unknown mechanisms, a strong active-automata/classical induction
baseline, and evidence that meaningful residual predictive ambiguity survives
matched fixed designs. Do not silently replace this finite prior with arbitrary
mutants or let the LLM merely classify supplied machines.

The next research decision is source selection for genuinely open-support,
text-native mechanism induction with informative adaptive interventions, not a
Luna depth sweep on these eight groups. This result strengthens that selection
criterion but does not meet the project's positive LLM-native BED goal.

## Accounting

Authenticated at 2026-09-10T00:02:24+01:00: credits245, usage222.308414519,
balance22.691585481, unchanged from the final September9 observation. The new
London-day ledger preserves this boundary and carries .04 unresolved exposure
conservatively; available daily allowance4.96. New model calls/cost: zero.
Automations remain paused and no cluster was used. Previous turn was no progress;
this turn completed a new source opportunity measurement and independent control
certificate. The full research goal remains active and unmet.
