# Strict executable proposal interface; no paid launch

## Sequencing decision pending

The previous turn made concrete progress by implementing and measuring an
executable-search baseline. It did not produce calibrated program beliefs or
qualify a broad-prior planning reference. Continuing numerical development alone
still leaves the LLM's usefulness untested in this new task.

An explicit user decision was requested: may a new proposal-only DeepCoder gate
spend up to$0.25 within the existing account-wide$5/day limit BEFORE a qualified
full-grammar planning reference? Approval was not received during this work.
Do not infer it from a goal continuation, a preselected option, or this document.
The existing numerical-reference-first order and all closed results remain in
force. No model calls or new scientific program outcomes were requested.

Even approval would authorize only preparing/freezing the complete small gate,
not bypassing schema, accounting, fresh model/catalog checks or semantic criteria.
A runner, prospective cases, paired controls, failure scoring and target sealing
are still required before any paid dispatch. No such paid runner is added here.

## Implemented interface

`environments/program_induction/proposals.py` has no transport or credential use.
It accepts public real input/output histories and exposes the full typed source
vocabulary and sampling law. It has no field for hidden programs, arbitrary task
text, future targets, priors supplied by an LLM or evaluator metadata.

History-aware and history-blind modes use identical instructions and vocabulary.
The blind prompt replaces only the history with an empty list, including no
history-length signal. A future numerical fitter must still use the same actual
history in both arms. The module cannot certify that a caller supplied only
real observations; the eventual runner must enforce that boundary.

Strict JSON permits1--8 programs, each with2--4 implicitly named SSA steps.
Every operation, lambda, argument scope and type is checked against the pinned
interpreter. Each step after the first consumes its predecessor, matching the
source grammar. No literals, arbitrary Python, fitted constants or proposed
weights are accepted. The whole batch fails before returning any program if
one item is malformed. No Markdown stripping, repairs, retries or valid-item
salvage is performed. Exact canonical syntax duplicates return only once.

Response text is bounded at32KiB. Histories contain at most4 observations and
the frozen two-list input law. ERROR is represented by JSON null and handled
by the existing bounded interpreter, not treated as a missing observation.
Compiled programs can be syntactically valid yet inconsistent or predictively
wrong; those are separate future measurements, never silently repaired here.

## Verification and limitations

34 focused proposal/search/rejection tests passed in1.46 seconds; lint passed.
Tests cover execution, duplicate syntax, lambda types, ERROR, blind invariance,
copied history, unknown fields, scope/type errors, duplicate JSON keys, nonfinite
values, response bounds and whole-batch rejection. Only constructed fixtures
were evaluated; no LLM response or new benchmark endpoint was opened.

This is reusable interface mechanics, not proof that an LLM will generate good
programs. It supplies no model probabilities, posterior guarantee, calibrated
mixture, paid authorization or positive non-myopic result. The desired final
planning/anticipatory-discovery result remains unproven.

Authenticated credits/usage remain245/220.376693994, consistent with the London
Sept8 ledger and zero spend. No cluster, protected runtime or automation change.
Full goal remains active and incomplete.
