# COPEx-Task L3 Interface Amendment

Recorded on 2026-07-16 after the registered live interface smoke and before the formal
run. No policy endpoint, selected action, accepted strategy, or arm comparison was
read.

The smoke passed all legality, shared-cell, width-compute, grid-d2-compute, and
zero-LLM rollout-scoring mechanics. It made seven requests, using 3,404 prompt tokens,
1,993 completion tokens, zero reasoning tokens, and `$0.00115052`. One strategy cell
used posterior ranks 4 and 5, was rejected, and validated after the single feedback
retry.

The prompt contradiction was mechanical: the frozen grammar permits ranks 0--3, but
the posterior summary displayed six entries labeled 0--5. The only repair is to show
the four addressable entries labeled 0--3. The grammar, executor, model, plans,
likelihood, scorer, arms, compute budgets, seed, endpoints, gate, and cost cap are
unchanged.

Linear request/cost projection from the smoke is approximately 2,100 requests and
`$0.35` for the formal run. The preregistered conservative projection remains `$0.60`,
below the `$1.50` hard run cap and remaining project authorization.
