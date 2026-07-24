# GOAL: non-myopic sequential BED where the LLM does the irreducible work

The RockSample StrategyEIG result is clean but the LLM's role is ornamental there:
exhaustive search is a runnable baseline, the enabling structure is geometric, and a
classical planner needs no LLM. Keep it as a SUPPORTING result ("the exact-verification
half works"), not the headline.

**The real objective**: a non-myopic BED result where the LLM owns the parts classical
BED cannot do — generating the hypothesis space and/or the likelihoods in a semantic
space too large to enumerate — and non-myopia still wins. The sharpest form:
**non-myopia over the LLM's own path-dependent belief dynamics.** In real BED-LLM the
belief state is a generative process (the LLM regenerates + filters hypotheses each
turn, imperfectly, path-dependently). So the value of a question includes how good a
belief state it induces next turn — coverage of the truth in the regenerated hypothesis
set — which 1-step EIG is blind to and which cannot exist in a classical fixed-support
system. That is non-myopia that is irreducibly about the LLM being the belief machinery.

Honest risk: this project's whole evidence base says non-myopia is fragile under LLM
noise. A wash is the likely outcome. But a small, honestly-controlled positive here is
worth more than a large one on a toy, because it is the only version where the LLM is
necessary. Chase it; keep going; the goal does not "complete."

**Near-term benchmark note (2026-07-24):** Try CA-BED's Detective Cases environment.
Its fixed five-suspect support can provide a cleaner semantic test of question
generation, LLM likelihood estimation, and depth-two planning than open-world support
regeneration. Treat it as valuable only if the main result is genuinely LLM-native:
the LLM must supply indispensable semantic inference that a classical enumerative
planner cannot replace, and non-myopia must beat paired myopic and matched-compute
controls. Do not let a larger result on an LLM-ornamental exact environment displace
this objective.

You have full autonomy. Decide, log a line in STATE.md, run it. Read whatever
literature helps. Pick whatever environments, models, and methods you think will
produce the strongest result. The only reason to pause is running low on budget — then
say so and propose what the next dollars buy.

Each loop, roughly: pick the single change most likely to strengthen the result →
engineer the instrument until smokes are clean (models, thinking, prompts, grammars,
hyperparameters — iterate freely, smokes are cheap) → measure once with honest paired
controls and endpoints written down before you look → write it into `results/nonmyopic/`
and the paper → update STATE.md → pick the next change.

What "stronger" can mean (your call which to chase): more environments (anything with
enabling/unlock structure where myopic scoring misleads — Rock-class, ρ-POMDP-family,
gated sensors, dynamic sources); harder settings (bigger maps, longer horizons, more
targets, noisier answers); better baselines to beat; deeper mechanism analysis; frontier
or thinking models; the model-aware-lookahead idea on 20Q machinery; robustness across
seeds/models; anything a reviewer would ask for next.

Keep the measurement honest or it's worthless: paired trajectories with common random
numbers, a compute-matched myopic/width control, a random-strategy control, endpoints
that don't saturate (entropy-AUC, truth-log-posterior), criteria fixed before looking.
That's not bureaucracy, it's the whole value.

Common sense: OpenRouter, key from `OPENROUTER_API_KEY` (never commit/log), backoff and
fail-closed. On 2026-07-24 the live balance was $66.292031753. Preserve at least $25
through Monday 2026-07-27: cap additional pre-Monday spend at $41.292031753, check the
live balance before every paid stage, project cost from smokes before larger runs, and
stop paid work if the live remainder reaches $25. Ledger spends in `EXPERIMENTS.md`;
use a small serving smoke before paid runs and swap models freely. Do not pursue the
OatML cluster until Hanshal explicitly re-enables it; use OpenRouter or zero-call local
work meanwhile. Commit and push code + results, keep tests green, and add a few
`STATE.md` lines each session.
