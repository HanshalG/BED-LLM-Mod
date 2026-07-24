# MovieLens Prospective Uncertainty-Enriched v5 Formal Result

Date: 2026-07-24

Status: passed every preregistered mechanism gate. A fresh target-blind
ranking-fidelity experiment is authorized; no policy/depth claim exists yet.

## Execution

- 48 fresh users screened in frozen order using profiles and metadata only;
- first 12 with maximum best-of-16 EIG at least `0.02` enrolled before outcomes;
- exact enrolled IDs: `318, 864, 459, 145, 21, 401, 236, 825, 764, 680, 653, 758`;
- four recorded-rating branches per enrolled user;
- exactly 192 requests and zero reasoning, retries, forced exits, parse errors, or
  runtime failures;
- all 48 branches had six non-copy generated profiles and eight-profile supports;
- cost `$1.03085638`; remaining balance `$19.807333683`.

## Frozen Gates

| Gate | Required | Observed |
|---|---:|---:|
| Prospective enrollment | 12 | 12 |
| Mean oracle held-out NLL improvement | >= 0.05 | **0.06121** |
| Users improving by at least 0.05 | >= 6/12 | **6/12** |
| Users with branch spread at least 0.10 | >= 6/12 | **7/12** |
| Mean immediate-EIG held-out NLL regret | >= 0.03 | **0.05871** |
| Users with immediate-EIG regret at least 0.05 | >= 4/12 | **6/12** |
| Enrolled users with maximum EIG at least 0.02 | 12/12 | **12/12** |

Global Spearman correlation between immediate EIG and negative realized branch NLL was
`0.28137`.

The initial derived artifact incorrectly reported `gate_failed` because the recomputed
conjunction included its own stale `all_pass=false` field. Every substantive gate was
already true. The reduction was fixed to exclude `all_pass`, tests were rerun, and only
the derived status was corrected; no model call, metric, threshold, sample, or endpoint
changed.

## Interpretation

This is the first clean external semantic mechanism pass in the project. The claim is
conditional and prospective: when the LLM's own generated semantic belief exhibits
measurable uncertainty before outcomes, different recorded queries induce different
regenerated profile supports, the best branch improves held-out prediction on average,
and current-support one-step EIG often selects the wrong branch.

The likelihood model never sees raw history, so branch prediction changes must flow
through the LLM-generated profile support. The result therefore establishes the
required path-dependent, load-bearing LLM belief opportunity. It does not yet show that
a non-myopic scorer can predict the better branch. The next authorized experiment is a
fresh ranking-fidelity gate, not a policy or depth sweep.

Artifacts:

- `results/nonmyopic/movielens_uncertainty_enriched_gate_v5/formal_seed24306_20260724/GATE.json`
- `results/nonmyopic/movielens_uncertainty_enriched_gate_v5/formal_seed24306_20260724/run.log`
