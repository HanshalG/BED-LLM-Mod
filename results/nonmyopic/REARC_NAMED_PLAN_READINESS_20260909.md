# Named-plan Luna-medium readiness

Fresh six-task cohort/source runner frozen at 42b8b145 before source execution.
All 18 source checks passed, without replacement. Cohort excludes all 26 earlier
tasks. Source artifact SHA256:
784385e87f68fef467382efe91a197f458e88bb96b00a51efbc057cca54c7a02.

One-shot public preflight passed: 66 source channels, six visible demo outputs,
60 withheld outputs still sealed. Public SHA256:
364149d4f008c00f257fe8f11d99862ddf13384eb54b6394772d40d294d8eec2.
All 24 plan/compile prompt checks passed; maximum 26690 serialized bytes.
The live repair request has its own enforced 65536-byte cap.

54 focused tests passed, covering shared Unicode schema, source channels,
paired 36-call controller, endpoint sealing, budget reservations, dependency
version mismatch, complete/null/interrupted zero-call replay. Closed mechanism
run still replays exactly with one old call and zero new calls.

Execute using /private/tmp/bed-plan-contract-env/bin/python. Runtime versions
must match requirements-plan-contract.txt; its hash is included in run bindings.
Exact model openai/gpt-5.6-luna, medium reasoning, OpenAI-only, max16384 completion
tokens. Reserve $0.08 before each attempt, full block ceiling $2.88. No retries.
Live account observed credits245/usage221.517287569/balance23.482712431;
conservative London Sep9 remaining3.90099091 before this block. Refresh before
dispatch. No model calls yet. All existing scientific gates remain unchanged;
this test qualifies predictive support, not non-myopic efficacy.
