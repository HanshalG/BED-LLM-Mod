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

**Resolved benchmark note (2026-07-24):** CA-BED Detective Cases was tried and
closed. Matched response-function likelihoods repaired depth-two truth-gain ranking
from negative to strongly positive, but only 23 of 241 questions (9.5%) distinguished
the murderer and innocent roles, only 4 of 12 cases were rankable, and depth two
changed/won on only 2 of 12. The released cases provide one canonical private story
per suspect rather than coherent counterfactual worlds, so richer prompting or a new
budget model cannot create the missing experimental opportunity. Do not reopen this
formulation unless the benchmark releases world-conditioned stories or response
models. Keep the headline on genuinely LLM-native path-dependent belief dynamics.

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
fail-closed. Enforce a hard account-wide `$5.00` cap per Europe/London calendar day.
Open each paid day from authenticated cumulative credits and usage, count unrelated
account use, reserve worst-case in-flight exposure before dispatch, and reconcile the
larger of posted and locally measured spend. Aim to put the allowance into the most
useful dependency-valid experiment available that day; do not borrow, roll over unused
allowance, count an unposted top-up, or invent an invalid experiment merely to fill the
cap. Check the live balance before every paid stage and use source/mechanics gates to
avoid experiments that cannot answer the research question. Ledger spends in
`EXPERIMENTS.md`; use a small serving smoke when interface risk is material and swap
models when a direct task gate supports it. Do not pursue the OatML cluster until
Hanshal explicitly re-enables it; use OpenRouter or zero-call local work meanwhile.
Commit and push code + results, keep tests green, and add a few `STATE.md` lines each
session.
