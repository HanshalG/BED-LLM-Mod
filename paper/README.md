# Validation-First Non-Myopic BED Draft

Current headline: in the Number Game, non-myopic planning over LLM-generated
belief trees beats myopic selection across four disjoint canonical-target
cohorts and two planning models by 12.48% Brier
(`95% CI [-.0192,-.0113]`; `95/128` wins), with positive first-link
calibration. A hash-bound retrospective audit gives 3.79% versus
compute-matched fixed-support depth three, but the preregistered
second-refresh mechanism test is null. A fresh dynamic-versus-fixed study
again favors dynamic support by 4.45%, but its confidence interval crosses
zero and it wins only 14/32 trees. The paper therefore claims a robust
LLM-native policy result, not a settled causal regeneration or monotonic-depth
result.

This directory holds the evidence-supported workshop draft on non-myopic BED
with LLM-derived probabilistic models. It combines exact planning controls,
the paired Rock Diagnosis policy result, the exact Gated Sensor qualification and
LLM-interface audit, exact UCI Mushroom, Cleveland, and Thyroid semantic-unlock
qualifications, the passed Thyroid 26B proposal gate, failed 26B/GPT trajectory
transfers, positive projected-utility confirmation, and projection-only factorial
ablation, a strict exact range-gated d3 opportunity where successor-grounded
26B recovered the optimum but a tied-control identity audit blocked trajectories,
an independently audited corner-start exact d4-over-d3 gain with failed
score-free h4 LLM serving gates and a hierarchical target-selection mechanism
smoke that recovered the route before the full serving gate failed, plus a
focused-prior exact h5-over-h4 qualification and audited hierarchical h5
proposal gate selecting the exact route on 15/16 cells, followed by a positive
50-pair cached hierarchical h5 trajectory confirmation and three fresh 50-pair
replications against shared h4, matched-random h5, and exhaustive d4,
the banked animals result, the natural-location depth audit, the
$\tau$-Knowledge target-blind root-ranking study, held-out count-dominant
receding-policy result, failed same-task execution replication, and the
ClariQ multisample-likelihood development signal with a serving-invalid
untouched holdout and a post hoc dynamic-support particle replay showing that
answer-conditioned future entropy can reduce first-link fidelity, an
InfoQuest shared-action causal mechanics result in which complete support
regeneration loses to compute-matched fixed support, plus a strict
semantic-partition exact-EIG replay and post hoc cached-answer diagnostic that
localize the loss to first-link question ranking rather than simulator noise,
and an all-candidate target-alignment audit showing abundant target headroom but
negative EIG-to-target correlation, a fresh Hotpot train opportunity gate with
320/1,000 strict directional unlocks and 31 qualifying top-four retrieval
misses followed by a clean serving pass and a no-repair confirmation failure
before scientific scoring, an
exploratory externally scored Zendo
path-dependent-belief smoke with pre-endpoint confirmation failures, and the
Collaborative Battleship released-bank opportunity followed by a target-blind
fresh executable-bank depth null, plus the Paprika and MediQ validation
failures. Endpoint-invalid Paprika outcomes
are diagnostic only, and no MediQ policy claim is made after the frozen likelihood
gate failed.

Draft validation:

```bash
python scripts/validate_paper_draft.py
```

The validator runs `pdflatex`, `bibtex`, and two final `pdflatex` passes in a
temporary build directory, then checks that the draft stays within the 4--6 page
workshop target and includes the required validity claims and figure.
