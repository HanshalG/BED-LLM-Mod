# SciConvBench LLM-Native BED Mechanics Result

## Verdict

The five-case mechanics audit fails the frozen prerequisite/dependency gate and
closes this SciConvBench construction before opportunity access or model calls.

SciConvBench provides strong scientific clarification tasks, an explicit
one-question-per-turn interaction rule, hidden complete specifications, and
component-level recovery rubrics. In the released form, however, those
components are fixed checklist facts. Asking one question reveals one fact but
does not change which hidden world is possible or which next component exists.

## Frozen Mechanics Cases

| Case | Hidden components | Structural finding |
| --- | ---: | --- |
| `fluids:case_001` | 3 | Flow rate, roughness, and temperature are fixed independent facts. The released policy specifies a static order, not answer-conditioned branches. |
| `foam:case_001` | 3 | Geometry, mesh, and time controls are fixed facts. Geometry and mesh have a weak semantic prerequisite relation, but the hidden mesh does not vary with the geometry answer. |
| `matToolUse:case_005` | 7 | File/copy/defect-construction instructions are an additive implementation checklist. |
| `solMech:case_001` | 8 | Units, material, boundary conditions, two segment geometries, load, and modulus are additive problem parameters. |
| `solToolUse:case_017` | 3 | Point-load location, magnitude, and application method are additive load specifications. |

Strong answer-conditioned parameter dependencies: `0 / 5`.
Static prerequisite-like relations: at most `1 / 5`.
The preregistered continuation gate required at least `3 / 5`.

## Transition And Endpoint Audit

The official runtime gives the user simulator the complete hidden requirement
and asks an LLM to answer each free-form clarification. The release does not
provide:

- alternative complete specifications for the same incomplete request;
- a structured question-to-component action mapper;
- exact component-conditioned response text;
- a likelihood or response table across candidate specifications;
- a reference clarification planner.

The `missing_entities` strings are useful external rubrics, but final
specification recovery is evaluated by an LLM judge. Exact string matching
would reject valid paraphrases, while using the planning model or another LLM
to map responses/specifications reintroduces the semantic endpoint dependence
that failed cross-family checks in InfoQuest and KnowU.

The official runner also hardcodes ten turns. The five mechanics cases hide
3, 3, 7, 8, and 3 components, so a policy can ask for every item without an
information-acquisition tradeoff. Some unrevealed corpus cases have more than
ten components, but component count alone creates truncation pressure rather
than non-myopic value.

## Decision

Do not inspect the 40 opportunity records, build an LLM policy, or use
OpenRouter for this exact construction. Reopen only if a new protocol supplies
counterfactual scientific specifications with answer-conditioned component
availability and an independently executable or structured endpoint.

The opportunity `40`, development `20`, and holdout `253` semantic values remain
uninspected. OpenRouter calls/cost: `0 / $0`. OatML jobs: `0`.
