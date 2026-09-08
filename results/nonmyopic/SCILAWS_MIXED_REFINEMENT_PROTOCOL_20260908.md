# Prospective mixed-family integration diagnostic

Freeze before executing scripts/scilaws_mixed_refinement_audit.py. Four fixed
synthetic histories (empty; action0,+.7; action0,-.7; action0,+.7 then action1,-.7),
two disagreeing scalar regression families with different target features,
coefficient means/precisions and noise scales. Exact constants live in fixture().
No source values, generated proposals or endpoint selection enter this diagnostic.

Exhaustively optimize both root actions and both adaptive continuation actions,
repeats allowed, depth2, at orders4/8/16/32/64. Use the horizon-aware correction;
no action or partial pruning, so every root value is exact for its numerical
rule. Preserve five seconds and100000 evaluated nodes per plan; bank limits,
never fill missing values. All20 combinations execute once, no case replacement.

Reference requires both32 and64 complete with maximum all-root difference<=1e-5.
Candidate4/8/16 requires this reference plus maximum all-root error<=1e-4 and
reference action regret<=1e-4, retaining the preceding accuracy scale. No
reference convergence means no candidate qualification, even for same actions.
Both are numerical diagnostics, not rigorous continuous error bounds.

This h2 test is a necessary stress check, not general h3 qualification or an
automatic order reduction. A pass authorizes neither source measurements nor
LLM calls. A failure closes deployment of that order on this evidence; preserve
the threshold. Next decisions must address the observed numerical issue without
tuning task outcomes, costs or hypothesis support to manufacture horizon gains.
