# Native Python compatibility and predictive scoring

The one-shot synthetic bank `rearc_python_compatibility_20260909` completed
successfully: five actual isolated container executions, four unavailable-module
AST checks, zero benchmark examples, zero model calls, and zero cost. Its bindings
record the exact contract, worker, runtime, and forecast sources. No containers
remained after completion.

Python 3.8 grammar is now checked on the host as well as in the container.
Postponed annotations allow ordinary `list[list[int]]` type hints without runtime
evaluation on Python 3.8. Iterator use and the five permitted standard-library
modules passed the saved examples. Two fresh containers returned the same
set-order-dependent result with fixed hash seed. This is a narrow repeatability
check, not proof of universal determinism or a security certification.

The native adapter reuses the existing fixed-slot predictive scorer. Canonical
AST identity removes formatting/comment duplicates without claiming semantic
equivalence. Programs that fit observations but fail future execution retain
their posterior mass as prediction failure. Invalid syntax raises rather than
silently salvaging a batch. All-invalid support predicts unit failure mass.

Focused verification: 11 tests passed in 0.16 seconds across
`test_rearc_python_runtime.py` and `test_rearc_python_forecast.py`.
The original runtime smoke remains an immutable record of its earlier source
version; this compatibility bank does not relabel that earlier execution.

Authenticated at 2026-09-09T21:30:02+01:00: credits 245, cumulative usage
221.761214289, balance 23.238785711. The existing September 9 London ledger has
opening usage 220.458278479 and recorded spend 1.34293581, exceeding the posted
delta 1.302935810. Its unresolved 0.04 reservation is retained. No new reservation
or paid request was opened.

Next dependency: shared-plan Python-versus-DSL controller and a fresh frozen
qualification protocol before another Luna-medium call. These mechanics establish
neither predictive accuracy nor non-myopic efficacy. Closed cohorts stay closed;
the scientific goal remains unmet.
