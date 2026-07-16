# Final Result: Non-Myopic Sequential BED With LLM Candidate Restriction

## Claim

In two paper-defined Rock Diagnosis environments, two-step **exact incremental EIG**
beats both shared-candidate one-step EIG and a candidate-call-matched one-step width
control when an LLM proposes only three legal actions per decision. The LLM is
non-thinking and proposes candidates only; transitions, likelihoods, posteriors, EIG,
and MAP decoding are exact programmatic components.

This is a controlled depth-over-width result for posterior information quality. It is
not a claim that short-horizon MAP accuracy improved, and the two replications are
separate Figure 4 map configurations within the same external Rock Diagnosis domain.

## Confirmatory Results

Positive values are paired final-posterior-entropy reductions
`H(control) - H(d2)` in nats. Each row uses a fresh seed, 30 paired trajectories,
eight actions, K=3, 10,000 paired percentile bootstrap resamples, common hidden
targets/observations, shared root candidate cells, and the same logical LLM candidate
call allocation for d2 and width.

| External map | d2 - shared d1 | 95% CI | W / T / L | d2 - matched width | 95% CI | W / T / L |
| --- | ---: | --- | --- | ---: | --- | --- |
| Figure 4 `3-6` | +0.2009 | [+0.0740, +0.3330] | 19 / 0 / 11 | +0.1903 | [+0.0666, +0.3167] | 19 / 0 / 11 |
| Figure 4 `5-7` | +0.5819 | [+0.5069, +0.6437] | 29 / 0 / 1 | +0.5372 | [+0.4572, +0.6047] | 29 / 0 / 1 |

Every confirmation passed all frozen mechanics: root cells shared, selected actions
legal, and width's candidate-call allocation equal to d2's virtual root-tree
allocation. The preceding fresh exact zero-LLM gates also passed on both maps:
`3-6` K=3 gives `+0.2270` / `+0.1565` nats over d1 / width, and `5-7` K=3 gives
`+0.0960` / `+0.0629`.

## Mechanism

The LLM-proposed root cells contain both immediate noisy checks and an enabling east
movement. On `3-6`, d2 chose `move-EAST` in all 30 confirmations while both controls
checked rock 1 in all 30. On `5-7`, d2 chose `move-EAST` in 28/30 trajectories, while
width checked rock 2 in all 30 and shared d1 chose an immediate check in every trial.
The future-sensing geometry, not simply more proposal calls, explains the result.

The truth-log-posterior deltas are also positive: `+0.1429` / `+0.0992` nats on `3-6`
and `+0.5210` / `+0.4923` on `5-7` versus d1 / width. The endpoint MAP deltas are
zero or negative, however, so posterior entropy and truth log probability are the
supported monotone decode-quality readouts at this horizon.

## LLM Interface And Cost

The strict parser rejected illegal left-edge `move-WEST` outputs and used exactly one
registered validation-feedback retry; there were no terminal cell failures or
programmatic action substitutions. Raw rejected attempts were 6/2,690 provider
requests for `3-6` and 51/2,705 for `5-7`; the latter is an interface caveat preserved
in full raw artifacts.

The two confirmations used 5,395 provider requests, 3,204,388 prompt tokens, 98,973
completion tokens, zero reasoning tokens, and `$0.34694344`. Including the failed
`3-6` interface attempt and both eight-trajectory pilots, the entire Rock Diagnosis
LLM program spent `$0.45851739`, while the global OpenRouter ledger remains below its
`$40` authorization.

## Scope

The source domain is Araya-Lopez, Buffet, and Thomas's Rock Diagnosis task:
[paper PDF](https://members.loria.fr/olivier.buffet/papiers/jfpda13-b.pdf). It is an
external, static-latent information task with a deterministic motion/sensor model and
an exact finite posterior. The result isolates the proposed mechanism cleanly, but
does not establish transfer to natural-language answerers, learned likelihoods, or
arbitrary LLM-generated query spaces. Those are follow-on questions, not conditions
silently smuggled into this claim.

## Reproduction

- Exact gates: `scripts/nonmyopic_rock_diagnosis_oracle.py` and the reports under
  `results/nonmyopic/rock_diagnosis_confirmation/` and
  `results/nonmyopic/rock_diagnosis_5_7_exact_confirmation/`.
- `3-6` confirmation: `results/nonmyopic/rock_diagnosis_llm_confirmation/20260715/`.
- `5-7` confirmation: `results/nonmyopic/rock_diagnosis_5_7_llm_confirmation/20260715/`.
- Frozen preregistrations and the spend/launch ledger: this directory and
  `EXPERIMENTS.md`.
