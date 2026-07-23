# Range-Gated RockSample[7,8] Depth-Three Preregistration

## Motivation

The existing exact Rock Diagnosis result establishes a horizon-two planning gain,
but ordinary distance-decaying sensing does not produce a strict endpoint-aligned
gain for horizon three over the strongest horizon-two policy. This fresh adaptation
asks a narrower question: can a two-action approach requirement make the additional
planning step load-bearing?

The standard Smith--Simmons RockSample[7,8] map and eight independent binary rock
types are unchanged. A check has accuracy 0.55 away from its rock and 0.95 at the
rock coordinate. Movement is deterministic and uninformative. This represents weak
remote assay followed by high-fidelity on-site specimen inspection.

An exploratory seed-24140 screen over multiple accuracy pairs and the standard maps
was used to choose the fixed law and task. No LLM response was requested. The formal
qualification below uses fresh seed 24141.

## Frozen Qualification

- 500 paired truth states, sampled independently from the uniform 256-state prior.
- 8 executed rounds on standard RockSample[7,8].
- Exhaustive receding-horizon d1, d2, and d3 policies over every legal action and
  every positive-probability observation branch.
- Terminal-history EIG is the planning score at each local horizon.
- Primary endpoint: paired d3-minus-d2 entropy-AUC gain.
- Corroboration: paired d3-minus-d2 truth-log-posterior-AUC gain.
- 10,000 paired bootstrap replicates; seed 24141.

The gate passes only if the 95% lower confidence bounds for both primary and
corroborating gains are strictly positive and all mechanics pass: common truths,
legal complete traces, d2 initially checks, d3 initially moves, and every d3 trace
reaches at least one on-site inspection. The stored trace must then pass an
independent replay/planner audit. Failure stops this line before any LLM proposal
interface is implemented.

## Conditional Next Stage

A pass authorizes only a separately preregistered, non-thinking proposal-policy
smoke and quality gate. The intended interface will ask for compact named three-step
plans; the LLM will not update beliefs or score rollouts. Exact code will verify plan
legality, posterior updates, and EIG, with exhaustive d2, compute-matched d2 width,
and matched-random proposal controls. No LLM policy endpoint is authorized here.
