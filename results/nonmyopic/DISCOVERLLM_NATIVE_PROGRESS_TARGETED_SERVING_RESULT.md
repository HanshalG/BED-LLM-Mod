# DiscoverLLM Targeted Native-Progress Serving Result

## Verdict

The targeted native-progress interface serves perfectly but **fails its
preregistered causal-link gate**. All eight semantic stages completed, every
expected cell parsed, and the explicitly contrastive dialog actions correctly
never advanced a root. However, neither generic artifact `R1` nor the four-way
hypothesis-targeted artifact `R2` advanced any of the four latent priority
worlds:

| Action | Root advances |
| --- | ---: |
| `D1` | `0 / 4` |
| `D2` | `0 / 4` |
| `R1` | `0 / 4` |
| `R2` | `0 / 4` |

The frozen gate required `R2 >= 2 / 4`. No policy score, mechanics run, or
released endpoint was opened.

## Integrity

- Run: `discoverllm-native-progress-targeted-serving-20260728T000000Z`
- Task: `technical_writing:artifact_352`
- Model: `openai/gpt-5.4`, temperature zero, no reasoning
- Logical requests / HTTP attempts / retries: `8 / 8 / 0`
- Prompt / completion / reasoning tokens: `43,159 / 3,963 / 0`
- Cost: `$0.1673425`
- Forced exits: `0`
- Parse counts:
  - actions: `4`
  - root transitions / feedback / tiers: `16 / 16 / 16`
  - follow-ups: `16`
  - follow-up transitions / feedback / tiers: `64 / 16 / 16`
- Private raw SHA-256:
  `c8e465332128868af3df750ba2bdb04d4bafabf5f46e7e30d661588266eae7a2`

The failure is scientific rather than syntactic. `R2` supplied one complete
LLM-written artifact alternative for each candidate priority and the evaluator
applied DiscoverLLM's native best-alternative rule. Even this targeted action
did not satisfy every leaf of the source root in any world.

## Decision

Close the DiscoverLLM native-progress cycle. Do not tune the transition
threshold, parser, task, or action wording and do not open the reserved
mechanics tasks or opportunity split. The released source contains useful
semantic intent trees, but this tested shared-action interface does not expose
an operative enabling transition for non-myopic BED.

Authenticated post-run OpenRouter balance: `$32.496637594`. There is no fixed
reserve. OpenRouter only; no OatML or Slurm.
