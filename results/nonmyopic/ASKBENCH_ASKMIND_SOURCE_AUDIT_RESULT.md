# AskBench AskMind Source Audit Result

## Decision

The released AskMind task does not pass the zero-call eligibility screen for a
native non-myopic BED result. Do not spend OpenRouter credit on an AskMind
policy or construct a headline BED claim from this release.

AskMind is a useful interactive clarification benchmark: each example exposes
an underspecified question to the tested model while keeping an original
question, reference answer, and explicit missing-information checklist hidden
inside the evaluator. The released data does not, however, define a belief over
alternative user intents or a counterfactual response model. Each degraded
prompt is paired with one original intent, and the simulated user response is a
fresh LLM generation conditioned on that one record.

## Frozen Source

- Repository: `https://github.com/jialeuuz/askbench`
- Commit: `f35da92feda34504f10413313554438e7abaeb08`
- Combined AskMind SHA-256:
  `406b9a48036374552d9e819c63d5c3ba25feea764c68f391e21c552f6221af82`
- AskMind BBH SHA-256:
  `f45e3fddaf1e1730ffe0dfe934e9d5c6c8cdb57edbdf2c30498e2d8140a43b9d`
- AskMind GPQA SHA-256:
  `bc02fb9a173dfe50b91654476eb49115d2613487a5fb5f9455d11ffa0e39603c`
- AskMind Math500 SHA-256:
  `2b1435904044019fd1ead39a42b1ad16b44a87d86a57eb19fd46cb7d48221fc9`
- AskMind MedQA SHA-256:
  `b5b1fb6098bb3d310c182f1278fcceae3a19ad91bdedb601ffabcdd4f547522f`

The audit inspected repository structure, data schemas and counts, construction
prompts, and evaluator control flow. It did not use an OpenRouter model.

## Released Structure

| Property | Count |
| --- | ---: |
| Combined AskMind records | 400 |
| Unique original questions | 400 |
| Unique degraded questions | 399 |
| Records with list-valued checkpoints | 400 |
| Checkpoints per record | 2--10 |
| Alternative latent intents per record | 0 |
| Exact question-conditioned response maps | 0 |
| Released semantic action mapper | 0 |
| Released reference planner | 0 |

The four source subsets contain 1,272 MedQA, 1,000 BBH, 367 Math500, and 187
GPQA records. Each row has one `degraded_question`, one `ori_question`, one
`expected_answer`, and one `required_points` list. Repeated degraded prompts are
too rare to induce a meaningful empirical intent prior, and the release does
not group them as alternative worlds.

## Why This Is Not Yet BED

The AskMind construction prompt removes or blurs facts from one complete
question and records those edits as checklist items. During evaluation:

1. the tested model emits a free-form clarification or final answer;
2. an LLM judge maps the turn to checklist coverage and final correctness;
3. if clarification continues, an LLM simulator sees the hidden original
   record and generates a natural-language response to the current question.

This supplies a realized dialogue but not the likelihood
`p(response | question, latent_intent)` across competing intents. Checklist
coverage is also principally additive: the release does not define
answer-conditioned availability, enabling actions, or a branch where resolving
one ambiguity changes which ambiguity can profitably be resolved next.

We could create candidate intent particles with an LLM and use the simulator to
answer under each particle. That would be a new synthetic environment whose
worlds, transition model, and belief support are all authored by our method,
rather than a native AskBench comparison. It would not solve the target
alignment and self-confirming-simulator problems already isolated in InfoQuest.

## Consequence

Close the released AskMind interface as the next paid route. Reopen only if the
benchmark releases grouped counterfactual intents with structured semantic
variables/actions/observations, or if a separately preregistered construction
first proves a target-blind greedy-versus-lookahead opportunity without model
calls.

OpenRouter calls/cost: `0 / $0`. OatML jobs: `0`.
