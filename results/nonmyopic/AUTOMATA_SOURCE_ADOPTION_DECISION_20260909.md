# Stateful protocol learning: retain for a bounded opportunity check

This is a new candidate, not an approved paid route or a claimed positive result.
The intended LLM contribution would be inducing executable protocol hypotheses
from public command/response traces and protocol descriptions. It must improve
over strong symbolic learning at a matched measurement budget. A finite bank of
supplied true machines plus classical planning would be only a supporting oracle
check, not evidence that the LLM is necessary or useful.

## Sources and why this differs

[Automata Wiki](https://automata.cs.ru.nl/) provides state-machine benchmarks
including real protocol and embedded-system models, under its stated MIT terms.
Its [table](https://automata.cs.ru.nl/Table) includes MQTT and QUIC models. These
offer stateful command/response semantics instead of image-to-program induction.
No benchmark model file, transition table or evaluation outcome was downloaded
in this audit; the corpus zip was deliberately not fetched.

[AALpy](https://github.com/DES-Lab/AALpy/tree/ca24a8a2ae224bec8e229753bc9f6a55b6297870)
is pinned in a no-checkout source clone at ca24a8a2ae224bec8e229753bc9f6a55b6297870.
Its README lists active and passive automata learners, so LLM-free learning is a
real competitor. The inspected license is MIT. Its SUL API distinguishes words
from individual steps and resets before a membership query. Random-walk
equivalence testing consumes additional system steps; PerfectKnowledgeEqOracle
compares against the true machine and must not be given to any learning policy.
These source contracts are not evidence of a measured BED horizon gain.

## Executed contract evidence

Only the hash-checked upstream SUL class was extracted and executed, without
package imports, network or model access. A deterministic synthetic fixture has
arm/probe commands; it is not a scientific task or a hand-picked efficacy example.

| Operation | Responses | Completed query count | Reported steps | Actual attempted steps | Resets |
|---|---|---:|---:|---:|---:|
|One word: arm, probe|ack, armed|1|2|2|1|
|Two words: arm; probe|ack; idle|2|2|2|2|
|Interrupted word: arm, fail|exception|0|0|2|1|

The last row is an accounting-interface warning, not an accusation of a library
bug: counters of completed calls are not a hard experimental budget authority.
An outer broker must count reset/step attempts before execution, including errors,
and route ALL learner/equivalence-test access through it. Cache reuse can be free
only for already acquired deterministic observations, with common access for all
policies; it must not accidentally reset or advance a live system.

Two focused tests pass, including rejecting an altered source before execution.
The canonical fixture result is AUTOMATA_QUERY_CONTRACT_AUDIT_20260909.json.
This does not establish a deployment wrapper, baseline integration or source-wide
correctness. No model calls, benchmark executions or paid authority.

## Next decision, before infrastructure expansion or model spend

Perform one bounded metadata/source-contract pass on compatible real protocol
families: resolve exact reset semantics, total/partial transition handling, common
alphabets and output encodings, family/version overlap, and lawful reuse. Freeze
the candidate family/variant selection before transition-value analysis. Exclude
live internet services: run published simulator models only.

Then prospectively measure opportunity under an explicit source-model prior,
with a fixed physical-command budget and a common reset cost convention. The
objective is prediction of withheld reset-start traces, not driving the device
into a rewarded state. Compare ordinary receding horizons, full-budget reference,
adaptive versus committed sequences, random and a strong word-level myopic
baseline charged for every command. Account for the distinction between planning
within one word and selecting multiple independently reset experiments. A gain
caused solely by giving longer words more measurements is invalid.

No automatic monotonicity claim follows from statefulness. Require useful
full-budget-minus-myopic headroom and disclose saturated cases. If a strong
sequence-level myopic policy already captures the advantage, reject this route
before a large implementation effort. No model or task replacement after results.

If that source gate passes, a fresh LLM predictive gate must use models not
enumerated or exposed to the LLM, with semantic descriptions that do not encode
the hidden transition table. Test meaningful predictive alternatives and
calibration, not only observed trace fit. Compare RPNI/active-learning baselines
using the same data and a bounded experiment oracle; exact truth-based equivalence
queries are prohibited. Include masked semantic-name controls to distinguish
useful prior knowledge from benchmark memorization. Dynamic regeneration still
requires actual-versus-simulated transition fidelity before depth efficacy.

## State and budget

Prior turn produced the scene zero-headroom audit. This turn identifies a new
measurement contract and verifies its key confounds without buying model calls.
Authenticated account unchanged at23:43London:245credits/222.308414519usage,
22.691585481balance; conservative dayremaining3.10986396. Existing ledger validated.
No cluster, automation, old endpoint, runtime or gate was altered. Full positive
LLM-native non-myopic BED goal remains unmet.
