# Source-qualified representation pool: ready for runner integration

The bounded sampling protocol, candidate manifest and implementation were
frozen and pushed at d8854f79 before source responses. Actual execution stopped
at six accepted tasks among the first seven candidates:

| Candidate index | Task | Source disposition |
|---|---|---|
| 0 | 855e0971 | All 11 channels verified |
| 1 | 4258a5f9 | All 11 channels verified |
| 2 | a64e4611 | Rejected: verify-phase ValueError at channel 2 |
| 3 | bd4472b8 | All 11 channels verified |
| 4 | be94b721 | All 11 channels verified |
| 5 | bc1d5164 | All 11 channels verified |
| 6 | 868de0fa | All 11 channels verified |

There were 69 source attempts: 66 accepted-task channels and three attempts on
the rejected task. No retries. Seventeen frozen candidates were not attempted
and remain reserved. This is the prospectively defined reference-valid
population, not a claim about the whole RE-ARC inventory.

The pool and each nested success/failure prefix replay exactly without new
source calls. The full scheduled manifest was independently compared with the
frozen candidate manifest and exact seeds. Terminal source result SHA256:
d9bdffc3d07bcc4a76794abf0fed0924284422c6d58440586b4b3c957b1f31f5.

All 60 retained query/target outputs remain sealed as hashes. The public bank
contains only six demonstration outputs and public input grids. Actual prompt
preflight passed 18 checks using maximum-length verbal-plan fields. The largest
complete request is 31230 bytes, below the 65536 cap. Repair bodies must still
be checked after actual model responses; this preflight does not guarantee
every conceivable response fits. Public preflight bank SHA256:
f0b0b14e92d514d145c39dc03ddba668d9c99bf9bcce4e975f66f48bfdfd390f.

Focused tests: 36 passed in 0.76 seconds, covering fixed candidate selection,
allowed rejection versus infrastructure abort, exhaustion, no work after six
successes, immutable one-shot directories, pool replay, isolated arm history,
equal call/slot budgets, observed-only repair and seal-before-label scoring.
No containers remain. No API calls, reservations or spend occurred.

Next: integrate the existing budgeted one-shot response bank with native Python
execution caching and this retained source cohort; test exact success/failure
replay, pin all bindings, then dispatch the frozen Luna-medium comparison if
the live London-day allowance permits. Use the already banked source inputs,
not regeneration. Retained cohort and source bindings must be checked before
any later authorized hidden-output regeneration. The source class initially
loads all 24 candidates, so the paid wrapper must explicitly bind its task list
to the six accepted tasks before endpoint access.

Previous goal turn made progress through actual source-failure evidence. This
turn made progress through a prospectively fixed screen and a real qualified
source pool. Neither establishes representation superiority or non-myopic
efficacy. The full goal remains unmet. Last authenticated London time
2026-09-09T21:39:28+01:00: credits245, usage221.761214289,
balance23.238785711; current daily ledger and its unresolved reservation unchanged.
