# AgentClinic dynamic-support terminal result

Date: 2026-08-13

Decision: **close the exact AgentClinic V1 construction**

## What passed

The pinned 120-case NEJM Extended population passed source admission. The six
frozen mechanics cases then passed the zero-call native-data preflight: each has
five unique answer choices, exactly one marked answer, no verbatim gold answer in
policy-visible text, and nonempty patient and physical-examination channels.

No OpenRouter call, model response, strategy score, or diagnosis endpoint was
opened.

## Decisive structural null

V1 fixed the policy-independent intake to the normalized `patient_info` field.
The released AgentClinic implementation assigns that same field to the patient
agent's complete private state:

1. `ScenarioNEJMExtended.patient_information()` returns `patient_info`.
2. `PatientAgent.reset()` sets its symptoms to that return value.
3. `PatientAgent.system_prompt()` labels those symptoms as all of the patient's
   information.

Consequently, under V1 a patient answer is generated from information already
visible to the policy, the selected action, dialogue history, and model noise. It
contains zero additional information about the latent case conditioned on those
variables. Any apparent update from that channel would measure responder noise or
hallucination, not experimental information.

This is enough to reject the registered mixed patient/test construction before
serving. It does **not** show that AgentClinic's test channel lacks information,
that a different reduced-intake task could not work, or that any planner failed.
Those questions were not evaluated. A reduced-intake successor would be a new
interface designed after observing mechanics content and is therefore not
authorized on this frozen population.

## Closure

- semantic-serving calls: not opened
- dynamic-support likelihoods: not opened
- depth-two and compute-matched-myopic scores: not opened
- diagnosis endpoints: sealed
- development, confirmation, and reserve cases: sealed
- OpenRouter calls/cost: `0` / `$0`
- OATML cluster use: none

The source gate remains a valid population audit. The paper may describe this
only as a pre-serving validation null, never as planner or policy evidence.
