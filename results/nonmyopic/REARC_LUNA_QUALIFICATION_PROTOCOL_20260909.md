# Frozen Luna-medium proposal qualification

Scope: four IDs in REARC_SOURCE_SCOPE_20260909.json, unchanged/no replacements.
Demonstration seeds31100..31102, target seeds31200..31207, source difficulty[0,1],
single attempt per seed with exact generator/verifier agreement and source runtime
limits. Any source failure stops the panel. This is a new program-proposal interface,
not a repeat of the closed numerical-correction or Number Game semantic endpoints.

All prompts see the same3demonstration inputs. Initial and blind calls see only
demo0output; aware calls see all3outputs. No held-out inputs/outputs, task IDs,
source descriptions, reference code or verifier feedback enter prompts. Use generic
DSL documentation and strict4graph format from rearc_proposal_interface.py.

Call order: initial for all4tasks; each must have at least one program exactly
matching demo0, else stop before refresh/held-out outcomes. Then aware/blind for
each task, alternating order by task index. Maximum12calls. Model exactly
openai/gpt-5.6-luna, medium reasoning, no fallback, OpenAI provider, seeds31300+i
initial and31400+i for both paired refresh calls. Max16,384completion tokens,
complete request<=32,768bytes; no format retries, seed retries or replacements.

Evaluate pools: initial4; initial4+aware4; initial4+blind4; symbolic search outputs.
All conditioned on the SAME complete3demonstrations. Uniform prior over unique
canonical programs, deterministic consistency likelihood, normalized survivors.
An empty posterior is an explicit unit failure forecast on every target, not a
prior reset or omitted task. Record execution errors and duplicate rates. The
symbolic control has its declared6step/32beam/5,000attempt-per-depth limits and
is first-order only; it is not an exhaustive/fullDSL necessity baseline.

Run every forecast on the8held-out inputs, seal and hash all predictions BEFORE
opening target outputs. Reconstructed target outputs must match their initially
sealed source hashes. Predictive scoring uses normalized whole-grid Brier and the
same30x30padded marginal Brier. Every task has equal weight; each target within a
task has equal weight. No subset or best-program scoring.

Qualification requires all of:
- Mean whole-grid Brier aware improvement over blind >=.01 absolute and >=10%
  relative, with at least2of4tasks improving by >=.01 absolute.
- Aware is nonworse than initial, blind and symbolic on BOTH mean scores.
- Complete source, response, prediction and receipt coverage; valid forecasts
  sealed before outcomes, exact replay; no missing or substituted task.

This small development gate is not a publishable efficacy test. Even a full pass
authorizes only prospective branch-transition fidelity and same-objective horizon
opportunity design, never an automatic depth or headline claim. A null stays null.

Budget: hard account-wide$5London-day ceiling. Block cap$.72, maximum$.06reserved
per HTTP attempt. Re-read authenticated credits and endpoint catalog before every
dispatch. For price ceilings prompt$.20/M, completion$1.20/M, cachewrite$.25/M,
even conservatively billing65,536input tokens plus16,384output tokens including
cachewrite remains below$.06. Require no additional positive-priced fee categories
not included in that exposure; otherwise stop and revise BEFORE calls. Price rises
can stop or reduce a block, not increase its cap. Reauthorize after reservation,
reconcile max(posted delta, accepted receipts), retain uncertain exposure after
ambiguous failures. Source .env with export semantics; never log credentials.

The dependency-injected panel logic is implemented and tested, but paid execution
is NOT authorized until the source collector, forecast-sealing replay and budget
transport are connected and their failure paths pass. No example generation or
HTTP call occurs by importing or testing this panel. Freeze all implementation
hashes at launch and bank the first terminal result/failure once.
