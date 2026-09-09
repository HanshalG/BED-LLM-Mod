# Frozen two-task numerical opportunity diagnostic

Use only the already-opened author-source development tasks 1 and 10 at revision
33e166058404fd5eec3ec0f5080df9befbe884a8, with the six verified source hashes from
author_strings_audit. No other tasks, model calls or arbitrary-input labels.

Fix roles by numeric example ID: e1 initial; e2..e7 six candidates; e8..e10 three
disjoint evaluation targets. Budget three nonrepeated unit-cost queries, terminal
half-multiclass Brier averaged over the same three targets. Do not permute roles,
change budget or select a different pair after measuring opportunity.

Declared finite grammar: exactly one or two consecutive string-to-string operations.
Operations are lower, upper, title, swapcase, strip, identity, reverse; ASCII letters,
digits, uppercase letters, lowercase letters, remove whitespace; prefix/drop-first/
suffix/drop-last for k=1..6; split on space/comma/hyphen/underscore/period/slash/at,
selecting index0/index1/last (missing index gives empty); whitespace first-word,
last-word, initials. No fitted constants, lookup tables, target-derived delimiters,
extra operations or program edits. Standard Python string/regex semantics.

Prior: uniform over lengths one and two, then uniform independent operation choices.
Condition only on the initial pair. Preserve syntactic multiplicity even when
behaviors coincide. Exhaust the entire declared grammar, not a time-censored prefix.
The grammar is a diagnostic reference, not the true distribution of human tasks.

Reuse ProgramReference and HorizonPlanner unchanged: ordinary receding h1/h2/h3
with budget3, fixed open-loop optimal B3, and random without replacement. Compute
exact expectation under the finite prior-conditioned reference. Whole-task120s and
existing per-plan5s/100000-node limits. Any empty support or search limit is an
incomplete/null task, not a completed partial reference. No rescue or increased cap.

Opportunity passes only if both tasks complete with nonincreasing h1/h2/h3 risk,
aggregate h1 and h2 are positive, successive aggregate reductions are at least5%,
and h3 strictly beats random and fixed open-loop in aggregate. This is a development
screen only, not a population efficacy test or proof of any LLM role.

Save reference forecast matrices/initial support before using later real labels.
Then separately replay each deterministic receding policy on the supplied real query
outputs. An unsupported actual answer terminates that policy with explicit abstention
loss1; no smoothing or new support. Score fixed endpoint outputs and retain all cases.
Real-path diagnostics cannot rescue the reference gate. No paid authority results
from this diagnostic alone; predictive and joint-transition checks remain necessary.
