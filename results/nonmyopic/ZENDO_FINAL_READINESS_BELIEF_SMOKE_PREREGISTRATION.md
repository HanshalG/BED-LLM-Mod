# Zendo Final-Readiness Belief Smoke Preregistration

Date frozen: 2026-07-25, before any response for this interface.

## Claim and scope

This is one development-only smoke on the previously untouched public Zendo rule
`phi`. It tests the first causal link needed for LLM-native non-myopic BED:
whether a target-blind model can choose a lower-immediate-information root because
the root produces better outcome-conditioned, LLM-regenerated executable beliefs
for a second experiment.

This is scientifically distinct from the closed
`zendo-path-dependent-belief-1` interface. It removes the endogenous
LLM-belief entropy objective and the LLM-generated experiment bank. A deterministic
target-blind scene compiler supplies experiments, exact EIG chooses continuations
inside each regenerated support, one blinded LLM scorer selects only the first
root, and the hidden measurement-only endpoint is behavioral agreement with the
official rule after two realized observations.

## Frozen source and split

- Official repository:
  `https://github.com/topwasu/doing-experiments-and-revising-rules`
- Commit: `af07590c4f4f617a79791e173460e5a4322b727f`
- `data/zendo_cases.json` SHA-256:
  `6440c543ff491af13606b79c57384281fae4e8bc205366e67be06d1c81fbacbc`
- Task: `phi`, unused by the prior paid Zendo smoke.
- The official rule is development-visible in source but hidden from all model
  prompts and from all scene/root/continuation selection until responses freeze.
- Seed: `24369`.

## Frozen interface

- Model: `openai/gpt-5.4`, temperature zero, explicit non-reasoning.
- Exactly 10 physical requests:
  1. one initial population of 12 unique executable rule ASTs;
  2. eight isolated refreshes for four roots times two hypothetical labels;
  3. one blinded four-root final-readiness score.
- No response repair, coercion, replacement, or scientific retry.
- OpenRouter only. OatML is not used.
- Projected cost `$0.25`; hard run cap `$0.75`; the global `$25` reserve remains
  protected.

The deterministic compiler creates 256 unique legal scenes without using the
hidden rule. It collapses scenes with identical prediction signatures under the
initial support and keeps the earliest maximum-EIG representative. It selects
four informative roots nearest `1.00`, `0.75`, `0.50`, and `0.25` times the
maximum available EIG, with signature novelty as the frozen tie-break.

For every root/outcome branch, GPT-5.4 regenerates 12 executable rules from the
complete hypothetical history. Exact code computes the posterior on that new
support and chooses its maximum-EIG continuation from the same 256-scene pool.
The final scorer sees both outcome branches, their model probabilities, semantic
rules and posterior weights, and exact continuation, but no hidden rule and no
root immediate-EIG value. It emits four integer readiness scores.

Only after the scorer response parses are actual root and continuation labels
revealed with the official `phi` predicate. The endpoint is posterior-weighted
behavioral agreement of the refreshed executable support with `phi` over the
frozen 512-random-plus-official audit bank after both observations.

## Controls

- **Myopic:** maximum initial-support one-step EIG over the same four roots.
- **Fixed-support depth two:** maximum exact two-step EIG while retaining the
  initial 12-rule support, with the same roots and 256-scene continuation pool.
- **Model-aware:** maximum blinded final-readiness score; exact continuations
  still use each branch's regenerated support.

The smoke does not support a population claim, significance test, or paper
headline. A pass only authorizes a separately preregistered multi-task
confirmation on fresh Zendo rules.

## Frozen gates

Mechanics must all pass:

- exactly 10 adapter requests and 10 HTTP attempts;
- zero retries, reasoning tokens, forced exits, repairs, and parse failures;
- at least six initial behavioral signatures;
- four distinct informative root signatures, each minority-label probability
  at least `0.10`;
- at least four distinct refreshed branch supports;
- every exact continuation has positive finite EIG;
- readiness scores vary and have a unique maximum;
- total cost at most `$0.75`.

Scientific gates must all pass:

- model root differs from myopic;
- model sacrifices at least `0.01` nats immediate EIG;
- realized weighted-agreement range across roots is at least `0.10`;
- model exceeds myopic realized weighted agreement by at least `0.05`;
- model exceeds fixed-support depth two by at least `0.03`;
- readiness-score versus realized-endpoint Spearman correlation is at least
  `0.30`.

No threshold, task, seed, parser, pool, or scorer repair will be made after
outcomes are observed. Failure closes this exact interface.
