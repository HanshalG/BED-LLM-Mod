# ForceBench is not a fresh source; the old contrast also changes objective

## Primary source check

[MDA v4, Sections 4.2, C and H](https://arxiv.org/html/2608.09696v4) describes
ForceBench as an interface wrapper over DiscoverPhysics. It uses noisy particle
trajectories, a force-function hypothesis and an ODE solver. The physical meaning
of control knobs is part of the discovery task. Its illustrated diagnostic
experiment establishes a mechanism distinction, not a comparison of planning
horizons under a common objective.

Consequently, do not present ForceBench as an independent replacement for the
DiscoverPhysics source already used here. Do not put hidden world names or the
true meaning of control knobs into a new public-context prompt. The new chemistry
moment bridge is not a compatible trajectory simulator merely because both tasks
have continuous parameters. A force-law study would need a genuinely distinct
prospective protocol; old hidden-source protocols remain closed.

## Local evidence correction

The fixed-initial branch replication recorded MSE3.049085 for modular B versus
3.481325 for D, a12.416% reduction, but3.048537 for same-root fixed-support B.
Thus the LLM support increment was null. These saved numbers are unchanged.

Read-only AST/source audit now verifies that selection_metrics in
scripts/discoverphysics_dark_matter_structured_replication.py chooses:

- myopic_root: maximum immediate_values (computed by immediate_eig), line360;
- lookahead_root: minimum internal_risks (retained terminal trajectory MSE), line364.

Source SHA256 b2284cbb911704a7e011cba9244dc9dbf697b06648e7f0dc5ca6082dc060ba7f.
The same objective pattern is present in grounded_policy and retained_support_replay.
The fixed-initial replication preserves source selection rather than establishing
a new same-objective myopic root.

The robust B-versus-D endpoint contrast therefore does NOT isolate depth. It changes
both objective and horizon. My commentary earlier in this turn called it a clear
non-myopic root-selection gain; the precise claim is a predictive advantage for
the complete B policy over the immediate-EIG-selected D policy. It is not proof
that terminal-risk planning beats myopic terminal-risk planning.

A numerical counterexample test shows why this matters: three hidden states with
prior(.45,.45,.1), targets(0,0,10), and two noiseless one-step queries. The higher-EIG
query leaves target risk above8, while the lower-EIG query gives risk0. Neither
looks ahead. Objective choice alone can create a large apparent planning gain.
This does not prove that our measured B-D gap is entirely an objective effect.

## Decision

Do not launch a renamed ForceBench replication or extend the chemistry adapter
to ODEs yet. The next useful zero-call check is same-objective myopic terminal
prediction risk on the saved initial model and original root actions, with all
old data/control choices fixed. Inspect cached prediction availability first;
do not regenerate an old physical endpoint or sample fresh LLM branches. Report
whether one-step risk already selects B, and whether any distinct horizon
advantage remains identifiable. This is retrospective diagnosis, not permission
to rescue the closed support claim or replace its endpoint thresholds.

Two focused tests pass in .09s; no simulator/model import or outcome computation.
The source audit changes the next action and corrects claim scope. Previous turn
was progress; current turn new source/code evidence. Goal remains active and
unachieved. New cost$0; authenticated balance23.693468061, conservative remaining
London-day allowance4.11174654. No cluster or automation changes.
