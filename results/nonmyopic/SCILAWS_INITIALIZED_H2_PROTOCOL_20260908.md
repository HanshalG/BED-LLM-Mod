# Initialized full-action depth-two refinement

Freeze before execution. All eight designs and zero/affine/quadratic initial
tables exactly as the initialized h1 audit, no source outcomes. Full eight-action
adaptive search, repeats allowed, all four posterior families, 64 targets.

Orders 4,8,16,32,64, each with one shared 5-second/100000-node plan budget.
No pruning, action filtering, per-root budget reset or case dropping. Record all
120 plans and failures. Orders32/64 must both complete with all root values and
max absolute discrepancy <=1e-5 to count as a converged reference. Candidates
4/8/16 require max root error and reference action regret <=1e-4. Incomplete or
unconverged references cannot qualify a candidate. Report all24 cases.

Same-family quadrature convergence is necessary numerical evidence, not an
independent accuracy proof. H1 agreement does not guarantee h2 accuracy. Neither
this diagnostic nor a pass authorizes h3/source/LLM execution or efficacy claims.
