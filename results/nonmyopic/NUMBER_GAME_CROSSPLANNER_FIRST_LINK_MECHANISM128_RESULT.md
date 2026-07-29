# Number Game Cross-Planner First-Link Mechanism-128 Result

Run: `number-game-crossplanner-first-link-mechanism128-20260729T080614Z`

Status: **retrospective cross-planner first-link summary**. This analysis
does not alter any source study's registered status.

## Myopic Comparison

Across four disjoint 32-tree blocks and two planning-generator families,
path-dependent depth three selects a different root from myopic EIG on
123/128 trees. On those changed-root trees:

- simulated-risk advantage over the myopic root: `0.02005`;
- realized exact-canonical Brier advantage: `0.01579`;
- realized-advantage stratified interval: `[0.01179,0.01990]`;
- wins/losses: `95/28`;
- score-to-realized-advantage Spearman: `0.423`;
- stratified Spearman interval: `[0.255,0.569]`.

The first link replicates independently by planner family:

- Qwen 3.7 Plus: 62 changes, 50/12 wins/losses, realized advantage
  `0.01476` (`[0.00968,0.02022]`), Spearman `0.407`
  (`[0.163,0.614]`);
- GPT-5.4 Mini: 61 changes, 45/16 wins/losses, realized advantage
  `0.01685` (`[0.01090,0.02307]`), Spearman `0.447`
  (`[0.184,0.660]`).

All four blocks have positive mean realized advantage and positive point
estimate correlations. The two family-level intervals are the relevant
replication evidence.

## Deeper Controls

Against fixed-support depth three, roots differ on 109/128 trees. Mean
realized advantage is `0.00777` (`[0.00402,0.01169]`) and Spearman is
`0.289` (`[0.096,0.467]`). The rank relationship is strong for GPT-5.4
Mini but imprecise for Qwen, so this control is heterogeneous by family.

Against cross-fitted depth two, roots differ on 76/128 trees. Mean realized
advantage is `0.00332` (`[-0.00050,0.00711]`) and Spearman is `0.021`
(`[-0.233,0.267]`). Both family-level correlation intervals cross zero.

The simulator is therefore calibrated to reject myopic roots across two
planner families. It is not calibrated well enough to order depth two and
depth three, preserving the monotonic-depth null.

## Scope

This uses the exact per-root canonical endpoint already present in every
fixed tree. It tests only the first link from simulated ranking to endpoint
utility. It introduces no online observation or posterior-execution noise
and does not test a second link.

Model calls and cost are zero. Public `RESULT.json` SHA-256:
`b964013f675b10d4739eab622e26c5aacdc65cb428d046355d102e6df92acff1`.
