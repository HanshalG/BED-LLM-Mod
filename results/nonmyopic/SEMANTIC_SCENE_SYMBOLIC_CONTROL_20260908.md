# Runnable structured-scene symbolic proposal control

Status: implementation verification, not semantic efficacy or planning opportunity.
Prior turn was concrete progress (executable rule/belief mechanics). This turn adds
the missing non-LLM competitor rather than treating a scorer slot as a baseline.

## Search contract

`environments/semantic_scene/symbolic.py` implements deterministic enumerative search
using the shared strict rule compiler and membership evaluator. Input is only the
real observed scene/label history and explicit work/output caps. No target scenes,
hidden rule, teacher counterexamples or model service enter the interface.

Finite grammar, increasing AST node count and stable within-size order:

- Atomic attribute counts (equal/at least/at most, thresholds 0 through 7).
- Negated counts, negated-attribute counts, and distinct-object same/different pairs.
- Counts of conjunction/disjunction of two atomic predicates.
- Conjunction/disjunction of two atomic counts.

There are exactly 38,808 syntactic candidates. The entire grammar is not the full
compiled language: higher compositions and more complex pair-role predicates are
not searched. Defaults evaluate 4,096 candidates and return at most four consistent
canonical programs. Every candidate in the authorized prefix is evaluated on every
history record; search does not stop upon finding an attractive proposal. Outputs
are ranked by node count then canonical JSON. The budget permits at most 50,000
candidates and 64 observations, with at most 32 returned proposals.

Records retain each evaluated program, node count, history mismatch count and
canonical-duplicate status. Counts, scene-evaluation work, elapsed time, normalized
history hash and full-grammar exhaustion are returned. Empty support is explicit;
failure to find a consistent rule in a prefix is not proof none exists. Contradictory
deterministic observations are rejected before search. There is no model call,
transport/retry, random seed selection or free posterior reset.

## Evidence

42 combined focused tests pass in 1.45s. They include actual observation-conditioned
search, flipped-label and empty-history controls, deterministic repeatability,
unmutated history, scene-order-invariant history identity, exact work counts,
compiler-to-search-to-belief integration, and complete 38,808-candidate grammar
exhaustion. These are mechanics fixtures, not held-out model-discovery outcomes.

## Scientific limitations to settle prospectively

This is a real but bounded enumerative baseline, not a PCFG posterior, a universal
symbolic optimizer, or automatically compute matched to an LLM. Prefix order and
grammar coverage matter. The LLM should not win merely by using productions denied
to search; any larger-language comparison must report that mismatch and include an
appropriately strengthened control. Defaults are implementation defaults, not frozen
scientific settings or a spending authorization.

Canonical syntax deduplication does not identify all logically equivalent programs.
Choosing uniform weights on returned syntax can overrepresent one concept. Decide
the explicit prior and semantic-multiplicity treatment before outcomes. A public
reference scene distribution could support prediction-diversity selection, but that
would be a new declared interface, not retroactive access to sealed test scenes.

The next work remains one prospectively defined complete opportunity panel with
this comparator and genuine equal-budget horizons, followed only on adequate
evidence by a held-out LLM proposal gate. No improvement is claimed from program
counts or passing tests. The overarching non-myopic, LLM-native result remains
unachieved. No inference spending or cluster use; automation remains paused.
