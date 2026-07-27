# DiscoverLLM Priority-World Tier Serving Result

## Verdict

The coarse ordinal-tier realistic-input gate **passes every frozen condition**.
All five semantic stages and all 40 expected objects/tier rows parsed exactly,
with no retry, reasoning token, forced exit, repair, or reissue.

Unlike the complete-permutation interface, tier assignments allow ties and
therefore support different posterior entropy profiles. This passes only the
serving gate and authorizes the separately preregistered three-task mechanics
smoke. It is not evidence that non-myopic planning wins.

## Frozen Run

- Run: `discoverllm-tier-serving-20260727T213000Z`
- Model: `openai/gpt-5.4`, temperature zero, no reasoning
- Logical requests / HTTP attempts / retries: `5 / 5 / 0`
- Prompt / completion / reasoning tokens: `23,735 / 1,646 / 0`
- Cost: `$0.0840275`
- Forced exits: `0`
- Private raw SHA-256:
  `a9cc96858137452a17c96d16b8bf42b43cb5d8960f258943c48d9b6103e73361`

| Stage | Expected | Parsed |
| --- | ---: | ---: |
| Root observations | 8 | 8 |
| Root tiers | 8 | 8 |
| Follow-up continuations | 8 | 8 |
| Follow-up observations | 8 | 8 |
| Follow-up tiers | 8 | 8 |

The likelihood scorer and continuation policy never received the truth map.
Released DiscoverLLM scores and winner labels were not read. The public result
contains no semantic content.

## Budget

The live provider endpoint reports `$33.121915094` remaining but may lag the
latest full local charge. The conservative post-run balance is
`$33.052742594`. There is no fixed reserve; spending is paced through Monday,
2026-08-03.

OpenRouter only. OatML jobs: `0`.
