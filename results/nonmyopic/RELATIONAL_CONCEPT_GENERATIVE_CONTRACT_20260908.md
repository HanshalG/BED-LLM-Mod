# New relational concept prior: construction contract

This defines a NEW synthetic active concept-learning task. It does not reproduce
INDUCTION's canonical data distribution, claim to implement its private generator,
or reopen the previous small-object semantic-scene study. The language uses the
upstream Concept Synth parser/evaluator at
c1f71f98623ab6e3513f820a8e52235d5b498694. No released formula bank is used.

## Declared Prior

Maximum grammar height4. At each positive-height node, choose atom/not/and/or/
exists/forall with integer weights4/1/2/2/2/2. Height0 is an atom. Atoms are
P(v), Q(v), R(v,w), S(v,w) for all scoped variables, plus equality between each
distinct unordered variable pair. Atoms are uniform; binary child order matters
in the generative law. Quantifiers bind fresh variables in x,y,z,u,v order.
Root scope contains x. Condition ONLY on the parsed formula having exactly free
variable x. A structural draw limit64 returns failure, never a replacement seed.
This bounded procedure defines the distribution including its failure event.

Retain tautologies, contradictions, vacuous quantifiers and redundant syntax.
Do not deduplicate sampled hypotheses and then silently assign a uniform prior:
different derivations can give the same function and their masses must add.
Exact tree-count calculations are syntax counts before structural conditioning,
NOT a count of distinct semantic mechanisms or proof of LLM necessity.

World size is uniform on7..13. Independently for each unary predicate draw its
density uniformly on[.2,.8]; binary density uniformly on[.1,.5]. Conditional on
densities, draw all ground atoms independently, including self-relations.
The world sampler has no concept argument or access to labels. It never rejects
a world because its target extension is empty, full, easy or hard. Concept/world
randomness uses separately domain-tagged SHA256-derived streams.

These weights/densities are declared modeling choices, not source-estimated
parameters. They must not be tuned after planning gains or semantic-degeneracy
outcomes. Public grammar and world law must be equally available to symbolic
search/sampling and LLM proposals. No private reference list may advantage either.
Finite sampling by a classical method is a necessary baseline, not disallowed
because the derivation count is large. Claim LLM usefulness only from matched
compute and held-out predictive evidence.

## Current Authorization

Implement/test the construction and an aggregate structural sample only.
The structural sample is seeds0..1023, no resampling, no world labels, no planning,
no paid model calls. Report string diversity, structural retry histogram and exact
unconditioned derivation count. These are engineering facts, not scientific gates.
Do not output sampled formulas or future evaluation seeds in policy artifacts.
The tiny manually constructed unit-test world is not an experiment outcome.

Before any episode or source-semantic screen, separately freeze query/target seed
domains, splits, noise, B, full menu, endpoint loss, no-filter degeneracy checks,
and resource/accuracy gates. A degenerate or low-opportunity distribution must be
reported in full and closed, not conditioned on favorable labels or depth gains.
No empirical strict-monotonicity promise follows from Bayesian conditioning.
Use the existing horizon planner; no new planning engine is needed.

This contract addresses the missing reproducible non-tiny SYNTACTIC distribution.
It does not yet address the more important semantic richness, useful LLM proposal,
non-myopic opportunity or anticipated-discovery requirements. The full goal is
unchanged and remains incomplete.
