# tau-Knowledge Gemma 4 26B Thinking Scorer Result

## Decision

The frozen 14-prompt smoke failed serving. The 140-prompt confirmation was not
run. This is not evidence for or against cross-model scorer efficacy.

## Serving Result

- Logical first-pass prompts completed: `14/14`.
- Physical requests: `22`, within the at-most-28 ceiling.
- Myopic root objects parsed: `2/2`.
- Full-tree non-myopic root objects parsed: `2/2`.
- Focused continuation objects parsed: `9/10`.
- Reasoning tokens: `53,448`.
- Forced exits: `9`.
- Forced-final requests and successes: `8/8`.
- Adapter-attributed cost: `$0.03252216`.
- Confirmation requests: `0`.

Eight empty/provider-notice length stops exercised the registered
reasoning-disabled continuation and returned valid final answers. The ninth
length stop returned a nonempty 21-character JSON prefix beginning
`{"followup_1_`. The repaired adapter intentionally continues only empty or
recognized provider-notice truncations, so this partial object was not
continued or repaired. The strict parser rejected it and the smoke failed
closed.

## Diagnostic Only

The nine valid focused rows had pairwise accuracy `.6538` over 13 comparable
pairs, selected an oracle-optimal continuation on `6/9`, and incurred total
regret 3 documents. Both myopic and both non-myopic root responses had
nonconstant scores.

These rows suggest the model was not obviously ranking at chance, but they
cannot satisfy the exact ten-row gate. Even treating the omitted row as
oracle-optimal would yield only the minimum `7/10`; its actual efficacy is
unknown. No partial-row normalization, forced continuation, larger budget,
prompt change, parser amendment, or replacement smoke follows.

## Interpretation

Gemma 4 26B A4B with thinking can parse the long semantic scorer inputs and
usually emit the required schemas, including through the repaired forced-final
path. It is not reliable under the frozen one-continuation interface because a
nonempty partial JSON length stop bypasses that path.

The supported paper claim remains GPT-5.4-specific: GPT-5.4 ranking is
test-retest stable on the frozen trees, GPT-5.4 Mini failed scorer efficacy,
and Claude, Gemini, and Gemma cross-model efficacy are unmeasured because
their serving interfaces failed first.

## Budget

- Adapter-attributed smoke cost: `$0.03252216`.
- Live OpenRouter balance afterward: `$48.355746126`.
- Balance above protected reserve: `$23.355746126`.
- OatML use: none.

## Artifacts

- Preregistration:
  `results/nonmyopic/TAU_KNOWLEDGE_GEMMA26B_THINKING_SCORER_PREREGISTRATION.md`
- Public failure artifact:
  `results/nonmyopic/tau_knowledge_gemma26b_thinking_scorer_smoke/`
- Private raw responses: stored outside git.
