# Frozen finite-prior relational opportunity pilot

New grammar law fixed in RELATIONAL_CONCEPT_GENERATIVE_CONTRACT_20260908.md;
implementation f89d8a2f. No changes to grammar, height, densities or filtering.
This authorizes zero-cost synthetic label computation, not paid LLM work.

Four panels indexed0..3. Each uses512 prior draws at concept seeds
1000000+1000*panel+i, i=0..511. Each has72 independent worlds at seeds
2000000+1000*panel+j, j=0..71; evaluate the distinguished object0. Worlds0..7
are the full eight-query menu; worlds8..71 are the fixed uniform prediction
targets. No response-dependent world selection. No noise. B=4 measurements,
with uninformative padding if all remaining queries are determined. Initial
history is empty. All draws retained, including identical functions and constants.

The reference population is the empirical512-draw law with equal mass PER DRAW,
not equal mass per distinct string/function. This is exact finite-prior planning,
not exact inference over the whole grammar and not an LLM proposal evaluation.
Keep this limitation explicit even if the opportunity gate passes.

Use the existing exact bitmask membership planner, preserving duplicate mass in
separate particle bits and restricting terminal Brier risk to fixed target columns.
Replan at every realized branch with min(h, remaining budget), h1/2/3. Controls:
receding open-loop horizon3, uniform random remaining query and exact one-step
optimizer (same values as h1; a saturated myopic reference, not a token-matched
LLM control). No empirical prior or policy may inspect which particle is true.
Expected endpoints integrate all response paths under the same finite prior.

Limits:120seconds likelihood construction per panel;5seconds shared comparison
work per panel,100000 planner states and100000 open-loop sets. Runtime is amortized
across cached policy comparisons, not an independent per-arm timing claim. Output
is reserved before outcomes and checkpoints every completed panel; preserve any
failure prefix, never rerun a banked result.

Report all four panels, exact rational losses and floating summaries, first
queries, constant-extension counts, unique extensions and likelihood hashes.
Reference opportunity requires positive h1/h2 risk, at least5% successive aggregate
h1-to-h2 and h2-to-h3 improvement, strict aggregate h3 advantage over receding
open-loop, and nonincreasing h1/h2/h3 in at least3/4 panels. A failure is a null,
not permission to retune the prior, remove constants or try new seeds. This is
not a population significance test, LLM advantage, or anticipated-discovery claim.

Even a pass authorizes no paid calls: next require independent population/prior
accuracy and useful executable LLM proposals with symbolic/history-blind controls.
Freeze runner and focused synthetic tests before opening these new label matrices.
