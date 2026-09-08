# End-to-end episode broker verified

Previous turn: concrete local isolation progress. This turn connects isolated
policy execution, durable measurement accounting and a bounded trusted evaluator
worker, then collects a fixed-length final predictive vector.

`environments/scilaws/episode.py` adds `BoundedPointBackend` and `run_episode`.
The worker command is evaluator-owned, not selected by policy output. Each point
and replicate runs in a time/output-bounded worker; persistent row reservations
remain in the broker's measurement ledger. A timeout therefore consumes reserved
exposure and halts the session rather than creating a replacement draw.

The policy receives only the caller's audited public task, fixed target points,
declared rounds/replicates and its own returned measurements. It returns exactly
a point per measurement turn and a finite fixed-length prediction vector at the
end. The broker owns request IDs and replicate counts. Duplicate JSON keys,
nonfinite constants, seed fields, early submissions and wrong final dimensions
are rejected. No endpoint labels enter this API.

Episode journals are created exclusively, flushed and synchronized after each
record. An atomic ownership claim on a fresh measurement ledger prevents a
different journal from silently reusing an existing arm's measurements. Failed
or interrupted episodes are not restartable through this API. The final journal
record is either completion or a failed-closed status; abrupt process death may
leave a nonterminal prefix, which is not a completed result.

## Verification

55 combined SciLaws tests pass in5.38s; lint passes. A real isolated policy and
separate trusted evaluator complete a two-observation synthetic linear episode:
the evaluator alone reads coefficient3 from its private directory; measurements
at .25 and .5 return .75 and1.5; the final prediction at .75 is2.25. The policy
uses returned history rather than the coefficient file. This is a constructed
transport check, not an empirical LLM or non-myopic result.

Additional tests cover evaluator timeout with charged failed exposure, invalid
policy responses with zero measurement calls, duplicate keys, a wrong final
vector, journal replay refusal and alternate-journal ownership rejection.
Existing file/network isolation and upstream synthetic runtime tests still pass
after sharing the bounded subprocess transport. Policy execution always adds
the sandbox; the trusted evaluator deliberately does not, so it can access its
private task state. Never pass untrusted policy commands to the evaluator API.

## Scientific boundary

Execution plumbing is now integrated. Do not add more generic infrastructure
as a substitute for the next source-grounded scientific decision. The next step
is to freeze the complete bounded panel selection and public-prior/noise/target
contract, verify actual support/licensing, and measure genuine planning
opportunity before any paid proposal gate. The 29 metadata candidates have not
passed that test. No closed route is reopened by these software tests.

The caller still owns trusted runtime bindings, public-only staging, predictor
implementation and final independent scoring. The broker does not authenticate
an arbitrary supplied runtime-binding label or supply a task loader. It is not
a complete scientific study, an atomic API-spend dispatcher, or a formal security
proof. No model client was added or invoked. Account/ledger revalidated at
usage220.376693994, balance24.623306006, zero spend. No benchmark outcomes or
states loaded; automation paused and full research goal unfinished.
