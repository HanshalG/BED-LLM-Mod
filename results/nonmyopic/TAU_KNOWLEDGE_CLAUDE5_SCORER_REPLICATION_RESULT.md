# tau-Knowledge Claude Sonnet 5 Scorer Replication Result

## Decision

The serving smoke failed closed, so the 20-task, 140-call confirmation was not
run. This is a serving-interface failure, not a negative scorer-efficacy result.

## Frozen Smoke

- Model: `anthropic/claude-sonnet-5` through OpenRouter.
- Frozen source: two public V3 smoke trees.
- Planned calls: 2 myopic root scorers, 2 non-myopic root scorers, and 10
  focused continuation scorers.
- Frozen maximum output: 2,048 tokens.
- Reasoning was not requested, and the protocol required zero reported
  reasoning tokens.

The two myopic root calls returned complete JSON and parsed successfully. Both
non-myopic full-tree calls reached the 2,048-token completion limit and exposed
empty answer text, so the strict parser correctly stopped before any focused
continuation call.

## Usage

- Physical requests: 4 of the planned 14.
- Prompt tokens: 37,381.
- Completion tokens: 6,663.
- Reported reasoning tokens: 2,274.
- Forced length exits: 2.
- Cost: `$0.141392`.
- Retries, repairs, or replacement responses: 0.
- Confirmation calls: 0.

The reported reasoning tokens independently fail the frozen zero-reasoning gate.
The empty non-myopic answer texts independently fail the parser and completion
gates. The private raw checkpoint has SHA-256
`fff9c0a04b7d97747de0f84bd33a9abddda1823ab1b42ccb7e5a3ab4604188a9`.

## Interpretation

Claude Sonnet 5 used hidden reasoning even though the adapter did not request
reasoning. The shorter myopic prompts completed, while both approximately
14,000-token non-myopic prompts spent the full output allowance without
returning visible JSON. Therefore no root-ranking, continuation-ranking, or
end-to-end transfer statistic is available.

Increasing the output budget, allowing reasoning, shortening the prompt, or
changing the parser would define a new post hoc interface protocol. None was
used to rescue this smoke. The original GPT-5.4 tau result and its nonsemantic
controls are unchanged.

## Artifacts

- Preregistration:
  `results/nonmyopic/TAU_KNOWLEDGE_CLAUDE5_SCORER_REPLICATION_PREREGISTRATION.md`
- Public fail-closed artifact:
  `results/nonmyopic/tau_knowledge_claude5_scorer_smoke/tau-knowledge-claude5-scorer-smoke-20260725T015342Z/SERVING_SMOKE_FAILURE.json`
- Private raw responses: stored outside git.
