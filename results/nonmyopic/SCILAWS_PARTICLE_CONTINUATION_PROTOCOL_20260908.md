# Fixed future-history screen

Require qualified initial-history tail panel SHA256
95e53384a587fa90cdbbf776eb22613eae1dd72ec3f615a6839fff201dbce588.
Keep the32-node composite rule unchanged, including all tails and512 joint Sobol
particles per family. Use all48 ordered initial fixtures, not selected winners.
For each, take action0 and condition on branch indices0,15,31 of that predictive
rule: lower extreme, lower central-adjacent node and upper extreme. Exactly144
future states. This is a screen, not full-tree coverage or probability-weighted
efficacy; it omits other actions, most first observations and all second histories.

At each future state compare all8 one-step action values (including repeat0)
against a new independent full-density ParticleReference. Exact raw Gaussian
likelihood conditioning, no redraw/refresh/pruning of particles. Reference shares
5s/100000evaluations across8 actions; correction separately shares5s/100000states/
64MiB. Charge32 predecessor branches per initial fixture as separate construction
work; do not claim this whole audit fits a single policy-decision budget.
Every root error and reference action regret must be<=1e-4; references retain
error+tail<=1e-7 and mass error<=1e-8. All144 states must pass the screen.

Exclusive task/seed shards preserve all numerical/budget failures. No retry,
threshold adjustment, new counts, node movement, source observations or paid
requests. Do not redo banked initial-history plans or reference values. Only a
passing screen motivates a bounded actualh2 workload; even that has separate
full-Bellman integration and total-runtime requirements. A failed screen blocks
using this fixed rule deeper and must be diagnosed without calling it a pass.
