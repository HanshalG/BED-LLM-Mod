# Full shared predictive access saturates the deterministic neuron table

Frozen151d557a before calculation, protocol09405d0b, original bank34ba78da.
Both conditions completed within .13s each, with7725/6818 value/random states,
below60s/250000 caps. No models or simulator were called. Old endpoint artifacts
and failed gates remain unchanged.

| Shared planning/prediction prior | h1 MSE | h2 MSE | h3 MSE | Exact random MSE |
|---|---:|---:|---:|---:|
| Uniform all22 executable models | 0 | 0 | 0 | .901967595 |
| Oracle uniform15 pair models | 0 | 0 | 0 | .594428068 |

Every policy gets four physical queries. Values are exact rational results;
zeros are not rounded small errors. All15evaluated truths have zero final loss
at every horizon. The planner uses precisely min(h,remaining) observations for
selection, executes one, and replans. Neither condition has a restricted forecast
support or an oracle proposer that gradually reveals models to the predictor.

## Independent trace check

Reconstructed each saved posterior from its initial pool and actual table
observations, checked no repeated actions, verified support sizes, recomputed
the mean28-target forecast and exact per-truth squared loss with Fraction. All90
saved final losses agreed. No planning or new simulation was rerun for this check.

With all22models, h1 isolates1world after query1,11more after query2 and the
remaining3 after query3. With oracle15models it isolates2 after query1 and13
after query2. This directly explains the zero final error without invoking a
small numerical tolerance. Deeper policies can postpone resolution under the
fixed smallest-action tie rule; their differing roots are not evidence of useful
depth when every final loss is tied.

Descriptive mean loss after each realized query, from the same saved traces:

| Prior | Horizon | Round1 | Round2 | Round3 | Round4 |
|---|---:|---:|---:|---:|---:|
| all22 | 1 | 4.792399 | .112500 | 0 | 0 |
| all22 | 2 | 12.283612 | 1.960615 | .799802 | 0 |
| all22 | 3 | 6.067870 | 5.993201 | 3.796627 | 0 |
| pairs15 | 1 | 4.409127 | 0 | 0 | 0 |
| pairs15 | 2 | 5.678730 | 5.601508 | 5.590397 | 0 |
| pairs15 | 3 | 12.824683 | 5.601508 | 5.590397 | 0 |

These post-hoc trajectory summaries are not a new AUC endpoint or gate. Do not
change the objective/tie rule or reduce the budget after seeing them to seek a
positive curve.

## What this rules out, and what it does not

There is no residual final-loss opportunity for h2/h3 over h1 under these full
deterministic predictive models at the declared budget. Do not buy a neuronal
proposer or depth sweep on the strength of the earlier large ladder numbers.

This is not a like-for-like ablation of the old experiment: we corrected the
horizon definition, shared predictor access, and used exact deterministic
conditioning instead of its Gaussian pseudo-update. We cannot attribute all
of the old loss to just one of those changes. The weaker justified conclusion
is that its positive-looking ladder did not establish the required ordinary
horizon opportunity against these coherent full-information controls.

The all22condition still uses a fully enumerated oracle family, not LLM-generated
open support. The pairs15condition explicitly knows the truth-family population.
Neither is a deployable LLM result. Stochastic channels, unknown continuous
parameters or a genuinely open mechanism population may behave differently,
but those need a new justified source prior and prospective protocol, not added
noise or missing support chosen to manufacture a gap. Old neuron route stays closed.

## Next scientific dependency

The recurring bottleneck is now sharper: a useful source must remain uncertain
under an honest full-access predictive model, while offering a semantic role for
the proposer. Avoid a new runtime/adapter until the source supplies that uncertainty.
The banked Number Game result remains the actual small ordinary-horizon signal;
its failed semantic/calibration routes remain closed. A new source-model study
must test predictive adequacy and residual horizon headroom jointly, with immediate
reuse of all models already computed. Do not enforce monotonicity by withholding
models from myopic prediction or by substituting policy improvement levels.

Three solver tests pass in .09s (exact horizon stopping, multicategory conditioning,
random averaging and cap failure), plus the prior horizon-characterization tests.
APIcost0; balance23.693468061, conservative London-day remaining4.11174654.
Previous/current turns progress; full goal remains unachieved.
