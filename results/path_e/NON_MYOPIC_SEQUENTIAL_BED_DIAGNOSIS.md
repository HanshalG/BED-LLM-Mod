# Why Non-Myopic Sequential BED Is Not Working Yet

Date: 2026-07-14

Status: **cross-environment method diagnosis and literature synthesis**. This report
does not reinterpret invalid endpoints as policy evidence and does not authorize a new
claims run. Its recommendation requires a fresh preregistration before more spend.

## Executive Verdict

The search algorithm is not the primary failure. The exact categorical depth-2
recursion is mathematically sound for the joint model it is given. The project fails
when that model is not a coherent generative model of the deployed interaction, or
when the environment contains no meaningful delayed-information advantage.

There are two separable requirements:

1. **Model validity:** the latent hypothesis must be sufficient to predict replies;
   the likelihood, posterior update, simulated branch, and deployed transition must
   describe the same process.
2. **Planning value:** the task must contain complementarity, gating, or another real
   reason that a lower-immediate-EIG question enables a better future experiment.

Our experiments have rarely satisfied both at once. When they do, in the exact
constrained location oracle, lookahead works. When only the first holds, greedy EIG is
already hard to improve. When the first fails, depth compounds and then maximizes model
error.

## Evidence Across The Project

| Environment | Model/endpoint validity | One-step result | Depth result | What it establishes |
|---|---|---|---|---|
| Exact Mastermind/location harness | Exact latent, likelihood, posterior, and simulator | Correct | Correct | The arithmetic and recursion can work. |
| Constrained exact location oracle | Exact model plus an explicit delayed-information trap | Greedy final RMSE 0.5606 | Depth-2 final RMSE 0.1529 over 120 trials | Lookahead has value when the model and structural gap are real. |
| Natural location finding | Exact physical likelihood, but LLM strategy proposals and noisy rollout ranking; naive already strong | Greedy/naive competitive | Depth was non-monotonic; constrained 30-trial d1/d3/d5 RMSE 0.929/1.127/0.542 versus EIG 0.412 and naive 0.166 | More horizon cannot repair poor strategy ranking or absent headroom. |
| Animals / 20 Questions | Concrete finite target and small answer space | EIG AUC 0.743 versus naive-thinking 0.465 | Paired d2-d1 AUC -0.040 on n=10; paired d3-d2 -0.126 on n=40 | BED transfer works; deeper planning is not established. |
| Paprika | Free-form cause/remedy support, action-changing world, invalid success attribution | EIG/arbitration did not beat controls on the invalid endpoint | Full2 2/10 versus EIG 3/10 at about 20x cost | Pure information gain is not task value, and the simulator/update mismatch is fatal. |
| MediQ iMEDQA | Official endpoint and replies are valid, but A-D often denotes treatment/mechanism rather than patient state; records are sparse | Not authorized: both likelihood gates failed or were inadequate | Not authorized | The remaining blocker is the probabilistic model, not code execution. |

The location ranking-fidelity gate independently supports this separation. Increasing
the exact replay budget from 256 to 1024 reduced score noise, but rank correlation with
the smooth realized target did not improve. Better Monte Carlo estimated the wrong or
weakly rankable quantity more precisely.

## What The Current Depth-2 Code Actually Optimizes

For a root query `x`, `FullTwoStepCategoricalEIG` computes

```text
Q2(x) = I(theta; Y1 | x, h)
      + E_Y1 [ max_x2 I(theta; Y2 | x2, h, x, Y1) ].
```

This is the right two-step total-information objective for a fixed coherent model. In
`methods/categorical_eig.py`, the implementation:

1. computes root EIG from the current prior and likelihood matrix;
2. branches over every positive-mass categorical outcome;
3. applies an exact Bayes reweighting on the fixed target support;
4. appends a synthetic branch observation;
5. asks the LLM for branch-specific follow-up candidates;
6. adds the outcome-weighted best follow-up EIG.

The failure enters through the supplied model and branch representation:

- The maximum over noisy follow-up scores has positive optimizer's-curse bias. With
  more candidates and branches, the selected continuation is increasingly the one
  whose model error is most favorable.
- Errors in the root likelihood alter both branch probabilities and branch posteriors;
  the next likelihood is then elicited under that already-wrong state.
- Synthetic branches contain only `Yes`, `No`, or `Unavailable`. Deployment receives a
  grounded fact sentence. Candidate generation therefore sees different information
  in planning and execution.
- Under data estimation, each action asks the LLM for a response marginal and two
  hypothetical A-D posteriors, then projects them into a locally coherent joint. The
  projection enforces one-step marginals but does not create one global generative
  process shared across actions and future histories.

If the true non-myopic advantage is `Delta` and the horizon-dependent model/ranking
error is `epsilon_H`, depth helps only when `Delta > epsilon_H`. Our exact constrained
oracle made `Delta` large and `epsilon_H` approximately zero. The LLM rollouts have the
opposite balance.

## The MediQ Failure From First Principles

### 1. The target is not a sufficient latent state

The implementation uses the exam label A-D as `theta`. That is a valid **decision
target**, but it is often not a valid **world state**. In iMEDQA, a correct option may
be a treatment priority, enzyme, mechanism, duration criterion, or diagnosis. Knowing
that the correct answer is "treat hypoperfusion first" does not determine whether the
patient has glucose above 250 mg/dL. Findings associated with multiple options can
coexist.

Therefore `p(reply | theta, query)` is not well-defined without an additional latent
patient record `z`. The needed model is

```text
p(z | initial evidence) p(reply | z, query),    theta = g(z),
```

and EIG should target `I(theta; reply)` after marginalizing profiles `z`. Directly
asking an LLM for `p(reply | correct option, query)` conflates exam-key semantics with
patient generation.

The failed factored-record replay demonstrates this empirically. Missingness became
exactly neutral, but available observations still reduced mean true-option log
probability by 0.118 nats; one glucose observation penalized the true treatment-priority
option by 1.243 nats.

### 2. The official information channel is too sparse

The held-out non-thinking naive bank produced 30 valid, grounded interactions, but only
8 were answerable from the official records. All other automated checks passed. This
is not a malformed-question problem: heavy menstrual bleeding, pica, rigidity, chronic
kidney disease, syncope, and similar discriminators were simply absent.

Under label-neutral missingness, the observed answerability rate gives the loose bound

```text
I(theta; response | query) <= (8/30) log(2) = 0.1848 nats/query.
```

This makes a depth-2 advantage intrinsically small. Making `Unavailable` label-dependent
would increase apparent EIG by learning which annotations tend to occur in records,
not which diagnosis is true.

### 3. The starting belief is already overconfident

The completed bank's decoder had mean top probability 0.867 after round 1 while mean
accuracy was 0.50; mean correct-option mass was 0.442 and mean log loss was 2.188. By
round 3, correct mass had fallen to 0.418. This is the raw in-context overconfidence
failure BED-LLM explicitly avoids with diverse hypothesis generation, filtering, and a
uniform distribution over retained hypotheses.

An overconfident wrong prior leaves little entropy for EIG to reduce and causes all
future branches to be evaluated around the wrong option.

### 4. Data estimation repairs algebra, not semantics

The current IPF projection is useful: it guarantees exact row/column marginals and
makes unavailable posterior-neutral. But it cannot make an unreliable hypothetical
A-D posterior meaningful. It also makes the elicited joint query- and current-prior-
dependent, so independently elicited nodes need not be marginals of one sequential
model. Algebraic coherence at one node is necessary, not sufficient.

## What Successful LLM Information-Seeking Work Has In Common

### BED-LLM

BED-LLM says the specification and update of the joint model are decisive. Its strong
setting has a concrete latent entity or complete user preference profile, a small
response space, a prior-likelihood construction, non-deterministic likelihoods, and
history-consistency filtering. It rejects raw in-context beliefs because they are
overconfident, and its data-estimation ablation underperforms prior-likelihood BED-LLM.

The paper also makes sufficiency explicit: a static likelihood is appropriate only
when `theta` captures everything needed to predict a reply. Otherwise the likelihood
must include history or the latent must be expanded. iMEDQA's A-D label fails that test.

### Uncertainty of Thoughts

UoT reports gains from depth 1 to 3, but it plans over an explicit possibility set and
uses an LLM to partition that same set into affirmative/negative subsets. Its medical
tasks have only 5 or 15 disease classes, use GPT-4 as the environment simulator, and
are simplified or curated for explicit diagnoses. This is much closer to entity
deduction than to arbitrary iMEDQA answer options.

UoT is useful evidence that depth can help when future possibility sets are stable. It
does not show that recursively eliciting incompatible probability tables over an
insufficient target will work.

### CA-BED

CA-BED also assumes an explicit finite set of mutually exclusive hypotheses, binary
questions, soft LLM likelihoods, and depth-2 lookahead. It improves over direct
prompting on 20 Questions and Detective Cases. However, the paper does not report a
depth-1 versus depth-2 causal ablation, so it does not isolate non-myopic value from
finite beliefs, likelihood scoring, or candidate generation. Its categorical-answer
extension helps 20 Questions but hurts Detective Cases, which the authors attribute to
outcome non-exclusivity and likelihood errors. That is directly relevant to MediQ.

### DAD and RL-BED

Deep Adaptive Design and RL-BED can optimize total sequential information rather than
a greedy proxy, but both rely on repeated trajectories from a meaningful simulator.
They amortize or optimize planning; they do not identify a missing latent state or
repair simulator misspecification. Training on the current MediQ model would learn to
exploit its annotation and calibration artifacts faster.

### Theoretical backdrop

When information gain is adaptively submodular, greedy policies are already near
optimal. A natural depth benefit requires non-submodular complementarity: a first
question changes which second question is available or useful. Unrestricted open-ended
question generation often removes this structure because a myopic policy can ask the
most diagnostic question immediately. We must demonstrate the gap, not infer it from
the word "sequential."

## A Viable Path To A Real Non-Myopic Test

The recommended next path stays inside the already authorized MediQ benchmark but
retires iMEDQA for method claims. The official iCRAFT-MD split is much better aligned:

- all 140 questions ask for the most likely diagnosis;
- records contain a mean 14.82 atomic facts (median 14, minimum 9), versus 11.20
  (median 10, minimum 2) for usable iMEDQA;
- `theta = diagnosis` is a concrete causal property of a patient, not a treatment or
  mechanism label.

This switch must be explicitly authorized and freshly preregistered because the frozen
iMEDQA gate has failed. It should not reuse or tune against the failed bank.

### Proposed model

1. Retain the exact four diagnosis options as the decode target `theta`.
2. Represent each option with several diverse concrete counterfactual patient profiles
   `z` consistent with the initial evidence and that diagnosis.
3. Filter profiles against observed history and place uniform mass over retained
   profiles, following BED-LLM's sample-filter-retain construction rather than raw A-D
   probability introspection.
4. Predict Yes/No from the concrete profile; use label-independent record missingness.
5. Compute EIG about the grouped diagnosis label after marginalizing profile-level
   outcomes.
6. Use the same canonical observation representation and the same profile filtering/
   update code in simulated branches and deployment.
7. Keep the scaffold non-thinking. A thinking LLM remains the naive baseline, per the
   experimental design; reasoning is not the first attempted repair.

The profile layer is not decorative. It is the missing random variable that makes
patient replies generative while preserving diagnosis as the scored target.

### Gates before any depth comparison

1. **Environment gate:** held-out FactSelect answerability at least 50%, all replies
   grounded/relevant, and no case selection based on outcomes.
2. **Prior gate:** profile-grouped diagnosis probabilities beat uniform on held-out log
   loss/Brier score and avoid high-confidence wrong collapse.
3. **Likelihood gate:** on a fresh policy-independent bank, available outcomes have
   positive mean true-label log gain; predicted EIG has positive association with
   realized entropy and true-label gain; missingness is exactly neutral.
4. **Branch equivalence gate:** for a fixed query/outcome/history, simulated and
   deployed updates produce the same support and probabilities. No category-only versus
   fact-text mismatch.
5. **Structural-gap gate:** before paid policy evaluation, show on held-out profile
   worlds that an exact depth-2 oracle beats exact greedy under the benchmark's query
   budget and candidate rules. If no gap exists, this is a one-step BED environment.
6. **Ranking-fidelity gate:** depth-2 estimated incremental value must correlate with
   oracle incremental value under shared candidates and common random numbers.
7. **Causal policy test:** only then run paired depth 1 versus depth 2 with identical
   root candidates, simulator seeds, stopping rules, and compute accounting.

No rollout-count sweep should precede these gates. More rollouts reduce Monte Carlo
variance; they do not reduce model bias.

## Paper Positioning

CA-BED now occupies the broad claim "lookahead plus explicit LLM likelihoods improves
closed-set information seeking." A defensible contribution here must be narrower and
more causal:

1. identify latent sufficiency and branch/deployment equivalence as prerequisites for
   non-myopic LLM BED;
2. provide the first paired depth-1 versus depth-2 ablation under a calibrated,
   profile-grounded clinical joint model;
3. report the exact harness positive result and the Paprika/iMEDQA boundary failures,
   rather than hiding them;
4. separate one-step BED transfer from the incremental value of horizon.

The strongest honest paper may be a characterization: **non-myopic BED helps only when
the LLM supplies semantic support inside a validated probabilistic state model; depth
cannot compensate for an insufficient latent or an uninformative interaction channel.**

## Decision

1. Stop iMEDQA scorer repair and do not launch Claim 1 or depth 2 on it.
2. Do not spend the remaining OpenRouter budget on more rollouts, deeper trees, or
   thinking-budget sweeps under the current model.
3. Ask for explicit authorization to preregister one iCRAFT-MD diagnosis-only profile
   gate inside MediQ.
4. If that gate cannot establish prior calibration, likelihood calibration, branch
   equivalence, and an oracle greedy gap, close the non-myopic empirical claim and write
   the cross-environment boundary result.

## Primary References

- Choudhury et al., ["BED-LLM: Intelligent Information Gathering with LLMs and
  Bayesian Experimental Design"](https://arxiv.org/abs/2508.21184), ICLR 2026.
- Li et al., ["MediQ: Question-Asking LLMs and a Benchmark for Reliable Interactive
  Clinical Reasoning"](https://arxiv.org/abs/2406.00922), NeurIPS 2024.
- Hu et al., ["Uncertainty of Thoughts: Uncertainty-Aware Planning Enhances Information
  Seeking in Large Language Models"](https://arxiv.org/abs/2402.03271), NeurIPS 2024.
- ["CA-BED: Conversation-Aware Bayesian Experimental Design"](https://arxiv.org/abs/2606.01182),
  arXiv:2606.01182.
- Foster et al., ["Deep Adaptive Design: Amortizing Sequential Bayesian Experimental
  Design"](https://proceedings.mlr.press/v139/foster21a.html), ICML 2021.
- Blau et al., ["Optimizing Sequential Experimental Design with Deep Reinforcement
  Learning"](https://proceedings.mlr.press/v162/blau22a.html), ICML 2022.
- Golovin and Krause, ["Adaptive Submodularity"](https://arxiv.org/abs/1003.3967),
  JAIR 2011.
- Sloman et al., ["Metrics for Bayesian Optimal Experiment Design under Model
  Misspecification"](https://arxiv.org/abs/2304.07949), arXiv:2304.07949.
