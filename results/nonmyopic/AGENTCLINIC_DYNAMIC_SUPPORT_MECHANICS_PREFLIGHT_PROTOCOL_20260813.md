# AgentClinic dynamic-support mechanics preflight protocol

Date frozen: 2026-08-13

Status: **zero-call native-interface preflight; no model or endpoint authority**

## Purpose

The AgentClinic source gate admits a released 120-case population, but it does not
show that the six frozen mechanics cases can support a diagnosis-blind patient/test
interaction. This preflight opens only those six mechanics rows and checks the
native data contract before any prompt, response, strategy score, or endpoint is
created.

Passing this preflight authorizes only a separately frozen semantic-serving
mechanics protocol. It does not authorize a model call by itself.

## Immutable inputs

- source protocol:
  `results/nonmyopic/AGENTCLINIC_DYNAMIC_SUPPORT_SOURCE_PROTOCOL_20260813.md`
- source manifest:
  `results/nonmyopic/agentclinic_dynamic_support_source/MANIFEST.json`
- source audit:
  `results/nonmyopic/agentclinic_dynamic_support_source/SOURCE_AUDIT.json`
- AgentClinic commit:
  `b6570edefb940857a7c334350656b29f9d984f24`
- data SHA-256:
  `d945305ee17ee1456053fbfe2e9d9c5e8b27d14538bf48ab8ace7306dc437b85`
- mechanics split ordered-list SHA-256:
  `3c60e70097899ade4d1a84a8055d42eea6d8c6240582fe36655660e755cb2f69`

The selector must reconstruct the value-blind split using the source-audit code.
No case replacement or hand selection is permitted.

## Exact gates

All six frozen mechanics cases must satisfy every gate:

1. The source and source artifacts match their immutable bindings and report a
   passing source audit.
2. The reconstructed mechanics split contains exactly six unique cases and its
   ordered-list hash matches the public source manifest.
3. Every answer list contains at least four nonempty, canonically unique choices
   and exactly one marked-correct choice.
4. The correct answer text does not occur verbatim after casefolding and
   whitespace normalization in the policy-visible question, patient history, or
   physical-exam text.
5. The fixed policy-independent intake is mechanically defined as the normalized
   `patient_info` field, with the gold answer list, image, physical-exam record,
   and diagnosis withheld. It is nonempty for all six cases.
6. Both native action channels are available: `patient_info` is the private
   grounding for patient questions and `physical_exams` is the private grounding
   for test requests. Neither channel is emitted by this audit.
7. No individual case ID, source text, image URL, answer choice, diagnosis, or
   endpoint value is serialized in a public artifact.

The fixed intake deliberately preserves the source text rather than asking an LLM
to summarize it. This prevents a hidden, policy-dependent model call before the
first design action. A later serving protocol may use a source-grounded target
model to answer atomic patient questions and test requests, but it must never
receive the marked answer or alternatives.

## Required successor before any paid call

A separately committed and pushed semantic-serving protocol must bind:

- exact planner and target model IDs, reasoning modes, seeds, prompts, schemas,
  token limits, concurrency, retry policy, catalog-price checks, and a stage cap;
- a doctor action grammar with only `patient_question` and `test_request`, one
  atomic request per action, and no proposed diagnosis in the action text;
- target prompts grounded only in the fixed intake plus the appropriate private
  patient or examination channel, never in answer choices or the gold diagnosis;
- at least four initial hypotheses and four candidate actions per case, with two
  actions of each kind;
- exact answer-obedience, repeated-response consistency, missing-information,
  non-leakage, differential truth-coverage, and branch support-regeneration gates;
- complete raw-response banking before any gold answer is opened;
- a frozen depth-two score, compute-matched myopic score, random control, and
  oracle-linked first-link gate on all six mechanics cases;
- zero endpoint access until all candidate rankings and selected actions are
  immutable.

Any semantic, schema, transport, budget, or ordering failure closes that exact
interface. It cannot be repaired on these cases.

## Accounting

- OpenRouter calls: `0`
- OpenRouter cost: `$0`
- OATML cluster use: none
