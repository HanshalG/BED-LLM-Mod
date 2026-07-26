# InfoQuest Information-Need Scoring Audit Result

## Status

The deterministic post hoc scoring audit completed exactly with zero LLM calls
and `$0` cost. It cannot rescue the preregistered V2 null.

Public audit SHA-256:
`f6ea36d545392d43c684cac32d550fe58cf066d9a33f1b6c9a49a106164a814e`.

## Protocol

The audit binds the V2 public/private mechanics artifacts and the frozen
target-alignment labels. It evaluates 13 target-blind aggregations of the
already-generated importance weights and action-resolution probabilities:
linear powers `1,2,3,4,6,8,12,16`, maximum resolution, weighted maximum,
resolution margin, maximum minus mean remainder, and peak ratio.

The power-one result exactly reproduces the banked V2 target correlation,
selected gain, and target-optimal count. No generated semantic text is emitted.

## Results

The preregistered linear score has target Spearman `-.1238`, selected gain
`.4333`, and 11/30 target-optimal choices. Increasing the power suppresses
diffuse low-confidence spillover: power 8 reaches `.1791` correlation and
`.6667` gain; power 12 reaches `.2787` and `.7333`; power 16 reaches `.3250`
and `.7333`.

Maximum resolution is the strongest tested variant:

- mean within-cell target Spearman: `.4454`;
- selected target gain: `.7667`;
- target-optimal choices: 22/30;
- positive fixtures versus fixed: 5/6;
- wins/ties/losses versus fixed: 14/13/3;
- mean oracle regret: `.3333`, versus `.6667` for linear scoring.

Maximum-minus-mean-remainder also meets every old threshold with `.2892`
correlation and `.7000` gain. Weighted maximum fails (`-.1232`, `.4333`),
showing that the gain is not merely caused by selecting one need: the model's
importance weights are part of the miscalibration.

## Interpretation

The compiler appears to identify direct semantic need/question matches, but its
moderate cross-need resolution probabilities and importance weights make a
linear expected-mass objective prefer broad questions. A direct-resolution
score filters that spillover and is the next prospectively testable mechanism.

Because maximum resolution was selected after inspecting these disclosed
outputs, all numbers above are development diagnostics. The next test must
freeze the max rule before generating beliefs or target labels on fresh
records, with the linear score and random action as paired controls.
