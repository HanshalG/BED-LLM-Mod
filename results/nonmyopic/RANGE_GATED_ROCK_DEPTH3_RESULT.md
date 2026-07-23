# Range-Gated RockSample[7,8] Exact Depth-Three Result

The preregistered exact qualification and independent audit both pass. On the
standard eight-rock map, weak remote checks have accuracy 0.55 and on-site checks
have accuracy 0.95. This creates a strict two-action approach requirement before a
high-fidelity observation becomes available.

## Frozen Endpoints

| Comparison | Entropy-AUC gain [95% CI] | Truth-log-AUC gain [95% CI] | W/T/L |
| --- | --- | --- | --- |
| Exact d3 minus exact d2 | +0.462960 [+0.459983,+0.465734] | +0.470138 [+0.450752,+0.489263] | 500/0/0 |
| Exact d3 minus exact d1 | +0.462960 [+0.459988,+0.465692] | +0.472044 [+0.452765,+0.491539] | 500/0/0 |

The independent audit uses separate planner recursion and bootstrap seeds. It gives
an entropy-AUC interval of [+0.459940,+0.465693] and a truth-log-AUC interval of
[+0.450613,+0.489091]. Every stored action is independently optimal, and every
position, seeded observation, posterior metric, and aggregate comparison replays.

## Mechanism

Exact d1 and d2 remain at the start and take weak checks. At d2, moving once cannot
reach a rock, so it gives up one weak observation without exposing the high-fidelity
channel. Exact d3 instead moves south twice from the standard start to rock 5 and
then checks on site. Every one of the 500 d3 trajectories initially moves and reaches
an on-site inspection; every d2 trajectory initially checks.

This removes the ambiguity in the earlier ordinary-sensor depth audit: the third
planning step changes the executed information-gathering policy and improves the
registered trajectory endpoint, rather than merely altering a terminal plan that is
later postponed.

## Scope

This is an exact structural positive with zero LLM calls. The range gate is a
deliberate diagnosis adaptation, not a claim about the reward-optimal RockSample
POMDP. It authorizes only a separately preregistered non-thinking proposal-policy
smoke and exact quality gate. A later LLM result must beat exhaustive d2,
compute-matched d2 width, and matched-random proposals under common truths; this
qualification alone is not an LLM-policy result.

Artifacts:

- `results/nonmyopic/range_gated_rock_depth3_qualification_20260723/REPORT.json`
- `results/nonmyopic/range_gated_rock_depth3_qualification_20260723/REPORT.md`
- `results/nonmyopic/range_gated_rock_depth3_qualification_audit_20260723/AUDIT.json`
- `results/nonmyopic/range_gated_rock_depth3_qualification_audit_20260723/AUDIT.md`
