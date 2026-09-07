# Source pilot freeze, before hidden worlds or observations

Protocol JSON SHA256:
`8e1fc9df41d177fa80b2e500c22c663e3a78a980ecedaa355c667116cc2e1d36`.

This is deliverable B's eight-world engineering pilot, not powered confirmation
and not LLM efficacy. Freeze the physical task before the preflight constructs
any source response matrix. The protocol's input order, public source versions,
seeds, designs, parameter law, noise and targets cannot be chosen after outcomes.

Use four source mechanisms: Michaelis-Menten, competitive inhibition, product
inhibition and ping-pong bisubstrate. Michaelis-Menten is the nominated lower-
opportunity comparison, not assumed to be empirically uninformative. Each has
four independent joint parameter particles from a log-uniform box spanning the
public easy v0/v1/v2 parameter values. Fixed parameters stay fixed. Uniform
structure priors and uniform within-structure particle weights give16 initial
particles. This is a coarse finite approximation with real parameter uncertainty,
not an exact posterior over continuous parameters and not a fitted true-world bank.

Eight hidden worlds are independently drawn, two per family, from the same public
parameter-box law using a distinct seed. They are not used to construct the
deployable prior or target predictions. A separately labelled population-oracle
diagnostic may later use their equal-weight true population, but no core policy
can access those particles. Numerical opportunity and calibration must be
reported separately for those two priors. No population oracle is opened by
the preflight.

All policies get three real measurements from the same four-design menu without
repeats. Three rather than four measurements preserves a nontrivial final
selection: with four of four designs every policy would collect the same set.
The four measurements in the earlier plan were feasibility suggestions, not a
previously frozen physical protocol. Every h3 run replans with min(3, remaining).

Observe a scalar log1p(primary source rate) plus independent Gaussian noise of
standard deviation0.15. This is an explicitly source-derived observation task,
not the official ChemBench multiplicative noise or official scoring metric.
Enzyme level, temperature and pH stay fixed at1,310,7. Source background effects
equal1 at these settings. Targets are64 fixed, equally weighted log1p rates,
sampled independently over the declared substrate/inhibitor/product box; selected
queries never remove targets. The same world/round/design-indexed noise table is
shared by all policies, including the separate oracle diagnostic.

Core arms: actual h1/h2/h3, open-loop-lookahead h3 with real replanning, adaptive-
integration refined myopic, and random. Refined myopic shares the same prior,
targets and maximum decision-time allowance; it uses that allowance for a more
accurate immediate objective, not filler computation. It is not a claim of
exact realized-FLOP matching. LLM compute-matched controls belong to deliverable C.

Numerical runtime budgets are60s/decision,300s/complete world and2400s/panel,
8M planner states,64MiB tensor allowance and16MiB outputs. Bank atomic per-arm
and per-world results. On the first incomplete world stop before the next world;
report the failure prefix, never a partial efficacy table or a shallower fallback.
No SQLite archive, detached duplicate run or automatic retry is permitted.

Prospective engineering continuation requirements: all8 worlds and all core
arms complete, mean h3 target MSE improves >=5% over h1 and >=5% over its prior
forecast, h3 is nonworse than h2, and h1/h3 differ on at least2 world trajectories.
Report every paired difference, all control results, calibration and any
simulation/refinement failures regardless of direction. These small-panel gates
only screen whether further development is worth attempting, never establish
statistical significance or authorize a headline. A null cannot be repaired
with a new seed, different noise/targets or a weaker threshold. The separate
new LLM semantic gate is still mandatory before any paid proposer experiment.

## First Executable Boundary

`pilot_data.build_public_pilot` constructs only public candidate parameters,
candidate predictions and common target coordinates. The separately named hidden
world constructor is not called by `chembench_pilot_preflight`. The preflight
verifies pinned source and passed synthetic bindings, then tests every root
integration rule under the full16-particle prior. It neither plans trajectories
nor exposes hidden parameters, realized measurements or target labels.

If too many posterior crossings exceed the candidate64-node rule, fail closed
and improve that numerical adapter before opening worlds. Do not reduce the
declared parameter support. Numerical implementation corrections before endpoint
opening must be tested and separately banked, with this physics protocol fixed.
