# Bounded private logical duplicate audit

Freeze before running. Re-read exactly the already pinned 375 FullObs records,
SHA55e21dee281be6ab779119dcce609d7d969e5528863c82a38fa4782fca0100ed.
Only private formula text is used; benchmark labels/facts are never consulted.
No formula strings, task identifiers, models or counterexamples may be emitted.
Use upstream parser at c1f71f98623ab6e3513f820a8e52235d5b498694, not a new parser.
Translate its AST to first-order Z3 expressions with common predicate functions,
free variable x and correctly scoped quantified variables. Z3 version4.15.3.0.

All unique normalized strings in sorted SHA order; each unordered pair once.
Ask satisfiability of their XOR, timeout20ms each, seed0. A proof of unsat means
equivalence on every nonempty domain, hence also the benchmark's finite domains.
A sat result establishes a distinction on some model, NOT necessarily a supported
7-13-object domain. Unknown stays unknown. Stop after90seconds of pair work and
bank the exact completed prefix; no repeated timeout attempts or exclusion of
difficult formulas. Parsing overhead is separate from the pair-work budget.

Report counts, proven-equivalence connected components and number of proven
equivalent pairs across the previous tentative split. Do not emit or seal a new
split, and do not claim this is a complete finite-domain equivalence procedure.
Even a fully classified run does not certify absence of restricted-domain overlap.
The audit is an information-leakage diagnostic, not a membership query experiment,
source-model execution or efficacy gate. It authorizes no paid/planning stage.

Prospective synthetic tests must cover alpha-renaming, Boolean reordering and
idempotence, negation/quantifier duality, shadowing, predicate/argument-order and
quantifier distinctions. Preserve a private-context-suppressed terminal failure
if source parsing fails; do not inspect formulas to tune the audit afterward.
