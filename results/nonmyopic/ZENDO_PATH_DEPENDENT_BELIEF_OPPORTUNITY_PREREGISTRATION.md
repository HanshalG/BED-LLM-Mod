# Zendo Path-Dependent Belief Opportunity Preregistration

Date frozen: 2026-07-25

## Claim Under Test

This is a development-only first-link test of the strongest project objective:
whether an LLM-generated hypothesis population has candidate-dependent future
quality after observing an experiment, and whether a two-step BED score that
simulates those population changes selects a better first experiment than
myopic EIG.

The LLM owns the natural-language hypothesis proposal and revision. A small,
safe JSON rule language makes each proposed hypothesis executable. The hidden
rule and all scientific endpoints are evaluated by a deterministic moderator
only after every hypothetical branch response has been frozen.

This is not a held-out policy result. The ten official Zendo rules are public
development tasks in this repository and their source has been inspected.
Passing this gate only authorizes a separately frozen holdout construction.

## Source And Fixed Tasks

- Paper: *Doing Experiments and Revising Rules with Natural Language and
  Probabilistic Reasoning* (NeurIPS 2024).
- Code: `https://github.com/topwasu/doing-experiments-and-revising-rules/`
- Pinned checkout: `af07590c4f4f617a79791e173460e5a4322b727f`
- Official cases SHA-256:
  `6440c543ff491af13606b79c57384281fae4e8bc205366e67be06d1c81fbacbc`
- Serving/mechanics smoke: `zeta` (red exists) and `xi` (blue touches red).
- Conditional opportunity stage: `phi`, `upsilon`, `iota`, `kappa`, `omega`,
  `mu`, `nu`, and `psi`.
- Selection and audit seed: `24349`.

The initial history is the first official positive example for each rule.
Prompts receive the scene and its positive label, but never the rule name,
rule text, hidden program, official test labels, or endpoint values.

## Frozen Interface

Model: `openai/gpt-5.4`, temperature zero, explicit non-reasoning, no
scientific retry or response repair.

For each task:

1. One call generates exactly 12 natural-language hypotheses with executable
   JSON rule ASTs.
2. One call generates exactly eight legal, distinct candidate scenes from the
   current particles. The first four are root candidates; all eight form the
   shared second-step bank.
3. Eight concurrent calls refresh exactly 12 hypotheses for each combination
   of the four roots and the two hypothetical outcomes.

Thus each task has exactly 10 logical and physical requests. Branch refreshes
receive the same current support and differ only in the appended
scene/outcome. They are instructed to preserve plausible particles and revise
contradicted ones. New support is reweighted from the complete branch history;
old probability mass is not aligned onto new particle identities.

The DSL permits recursive Boolean rule composition plus block predicates,
counts, universal/existential quantification, same-attribute rules, touching,
stacking, and largest-block rules. No generated Python is executed.

## Fixed Scoring

- Uniform prior over each generated 12-particle support.
- Soft deterministic observation model: `.95` when a hypothesis predicts the
  observed label and `.05` otherwise.
- Myopic score: one-step EIG under the initial support.
- Fixed-support depth two: immediate EIG plus expected best second EIG after a
  Bayesian update on the unchanged support.
- Model-aware depth two: immediate EIG plus expected best second EIG after the
  corresponding LLM-refreshed support.
- Ties are resolved by lower candidate index.

After all branch outputs are frozen, each generated rule is compared with the
hidden executable rule on a deterministic 512-scene audit bank plus the
official initial/test scenes. The principal external endpoint for a branch is
posterior-weighted behavioral agreement with the hidden rule. Also record
maximum agreement and whether any particle reaches `.95` agreement.

For each root, the realized branch is selected by the hidden moderator's
actual outcome. The LLM never sees that outcome during hypothetical generation
beyond the already frozen branch-specific yes/no intervention.

## Smoke Gates

The two-task smoke passes only if all are true:

- exactly 20 physical requests and 20 HTTP attempts;
- zero transport retries, reasoning tokens, and forced exits;
- every response parses without repair;
- every task has 12 valid initial particles, eight legal distinct scenes, and
  8x12 valid refreshed particles;
- each initial support has at least six distinct audit prediction signatures;
- each task has at least three informative root candidates;
- refreshed support signatures differ across at least four of eight branches
  per task;
- total adapter cost is at most `$0.75`.

Failure closes this interface before the eight-task stage. Formatting failure
does not authorize parser coercion, reissue, prompt repair, or replacement
smoke tasks.

## Conditional Opportunity Gates

If and only if the smoke passes, run the other eight public development tasks
with the identical interface. The opportunity stage passes only if:

- exactly 80 physical requests/HTTP attempts and all smoke-style mechanics
  pass, with cost at most `$2.00`;
- initial posterior-weighted truth agreement is below `.95` on at least four
  tasks, avoiding a saturated support test;
- the range of realized root endpoint values is at least `.10` on at least
  four tasks;
- some realized branch improves maximum truth agreement by at least `.10`
  over the initial support on at least four tasks;
- model-aware depth two beats myopic on realized weighted truth agreement on
  at least three tasks, loses on at most one, and has mean gain at least `.05`;
- model-aware depth two beats fixed-support depth two on at least three tasks,
  loses on at most one, and has mean gain at least `.03`;
- mean within-task Spearman correlation between model-aware depth-two scores
  and realized weighted truth agreement is at least `.30`.

These are conjunctive gates. A null closes this precise particle/DSL/interface
construction. It does not invalidate the source paper's online particle
revision result, which used a different model and evaluated final rule
prediction rather than non-myopic first-action ranking.

## Budget And Scheduling

- Verified live OpenRouter balance before implementation: `$46.491449126`.
- Protected runway through Monday: `$25`.
- Smoke projected cost: `$0.30`, hard cap `$0.75`.
- Conditional opportunity projected cost: `$1.20`, hard cap `$2.00`.
- No OatML submission or dependency. No cluster job is part of this gate.
- Raw responses are private/untracked; public artifacts contain parsed rules,
  scenes, diagnostics, hashes, and complete usage accounting.
