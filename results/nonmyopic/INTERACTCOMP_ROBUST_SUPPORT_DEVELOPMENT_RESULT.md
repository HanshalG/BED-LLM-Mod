# InteractComp Robust-Support Development Result

Date: 2026-07-25

## Outcome

**The frozen additive robust-support score failed its open-task gates. It does
not authorize fresh validation in that form.**

The run completed exactly and the auxiliary LLM population successfully
recovered missing answers. The failure was acquisition: adding model-identity
information to ordinary EIG did not overcome a large but misleading
within-support entropy term on the collapsed task.

## Frozen Results

| Metric | Task 76 | Task 38 | Gate |
|---|---:|---:|---:|
| Unique auxiliary entities | 3/8 | 6/8 | at least 4 each |
| Exact target particles in auxiliary support | 6/8 | 2/8 | descriptive |
| Ordinary-EIG selected root | 2 | 3 | control |
| Robust selected root | 2 | 3 | descriptive |
| Oracle root | 2 | 1 | robust equals oracle each |
| Ordinary-EIG endpoint rho | `+.8165` | `-.5443` | control |
| Robust-score endpoint rho | `+.8165` | `-.2582` | positive each |
| Robust endpoint gain over EIG | `.000` | `.000` | mean at least `.0625` |

Mean robust-score endpoint Spearman was `+.27915`, above the prior ordinary-EIG
mean of `+.13608` but below the frozen `.50` gate. The robust score selected one
of two oracle roots and produced zero mean endpoint gain over ordinary EIG.

On task 38, the truth-bearing consensus-check root received `.21576` nats of
model-misspecification information, the largest such term among its four roots.
However, its ordinary EIG was zero. The selected wrong root retained `.56234`
nats of ordinary EIG and received `.06220` misspecification information, so the
frozen additive score remained larger (`.62453` versus `.21576`).

## Interpretation

The target-blind auxiliary-support mechanism worked better than its diversity
gate suggests. Repeated target particles are meaningful belief mass, not a
serving collapse: six of eight auxiliary particles on task 76 were the exact
missing target. Task 38's auxiliary population also recovered the missing target
twice, although several semantic aliases of the current wrong interpretation
escaped exact-name exclusion.

The additive rule conflated two belief-state regimes:

- with a diverse current support, ordinary EIG is an identification objective;
- with a collapsed current support, ordinary EIG measures distinctions within a
  model already suspected to be wrong, while model-identity information is the
  relevant model-criticism objective.

The open-task evidence motivates a state-aware exploration/verification switch:
use model-identity information alone when at most half of the eight current
particles are unique, and ordinary EIG otherwise. This rule was formulated
after seeing the development endpoints and cannot be credited on tasks 76/38.
It requires a fully prospective test on untouched encrypted tasks. A fresh test
must also use semantic outside-support validation rather than exact-name
exclusion.

No coefficient, mixture weight, or alternate additive formula will be evaluated
on these returned auxiliary responses. The exact V1 prompt/formula is closed.

## Integrity And Cost

- Preregistered commit: `2925515`.
- Run ID: `interactcomp-robust-support-development-20260725T104000Z`.
- Model: `openai/gpt-5.4-mini`, non-thinking.
- Requests/attempts: `32/32`.
- Retries/reasoning tokens/forced exits: `0/0/0`.
- Cost: `$0.02193525`.
- Public artifact SHA-256:
  `1e9dbb701793a63d85222ec276cbde4c64840423ce5510fcabd6a3ce1e2c6ab0`.
- Private raw SHA-256:
  `73f1ba84ffa1b9d6e8ed7ffe129d13f4aa37971851a9ae3c11a4c2cd6fa05f69`.
- Project-ledger spend after the run: `$86.27320081920747`.
- Monday local allowance remaining: `$14.870103999999941`.
- Authenticated OpenRouter remaining: `$44.111601884`, or `$19.111601884`
  above the protected `$25` reserve.
- OatML resources used: none.
