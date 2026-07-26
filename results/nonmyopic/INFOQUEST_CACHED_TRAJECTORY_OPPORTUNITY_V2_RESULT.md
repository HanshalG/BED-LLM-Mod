# InfoQuest Cached-Trajectory Opportunity V2 Result

## Verdict

The fresh-split V2 cached-trajectory audit **passes every frozen structural,
missingness, and substantive gate**. InfoQuest provides a strong
path-dependent sequential opportunity for an LLM-native ranking-fidelity
experiment.

This is not causal policy evidence. Released trajectories contain one realized
future each and cannot show that a proposed non-myopic policy beats a myopic
control.

## Results

| Metric | Observed | Gate |
| --- | ---: | ---: |
| Trajectories | 480 | exact |
| Task/world cells | 160 | exact |
| Multi-turn | `1.0000` | `>= .95` |
| At least two initially unresolved | `.9979` | `>= .80` |
| Delayed gain at least two | `.9979` | `>= .75` |
| Mean initial reward | `.3354 / 5` | diagnostic |
| Mean final reward | `4.6771 / 5` | diagnostic |
| Mean delayed gain | `4.3417 / 5` | `>= 2.0` |
| Full completion | `.7792` | diagnostic |
| Mean turns | `15.8479` | diagnostic |
| Immediate-minus-shifted uptake / transition | `1.2356` | `>= .25` |
| Positive uptake advantage / trajectory | `.9875` | `>= .60` |
| Turn-count range at least two / task-world | `.8438` | `>= .40` |
| Three distinct first actions / task-world | `1.0000` | diagnostic |
| Final reward varies across runs / task-world | `.3500` | diagnostic |
| Empty later messages | `0 / 14,254` | `<= .005` |
| Trajectories affected by empties | `0 / 480` | `<= .02` |

The immediate next-policy messages reuse 10,679 newly revealed content tokens,
versus 1,873 under the frozen circularly shifted-answer control, across 7,127
transitions. Delayed checklist progress is nearly universal, yet 22.1% of
trajectories still fail to complete all five items within 30 turns. This avoids
the terminal saturation problem that weakened several earlier environments.

## Polarity Replay

The first V2 artifact had every scientific gate true but a false top-level
verdict because the redaction condition was stored as
`semantic_content_emitted: false` inside an `all(...)` dictionary. The original
artifact was preserved at SHA-256
`ca24275121e366314719155aa97fb072f8e862e7c9fe2d27e97be6c3dcb69015`.

The preregistered deterministic correction changed only that gate to
`semantic_content_not_emitted: true` and the resulting top-level `passed`
field. Canonical comparison confirms every other field is identical. Corrected
artifact SHA-256:
`4e20a793a1350ab5aabd0e6f9ebe8240c1519f492d462d8320e51ee8783cab28`.

## Interpretation

InfoQuest clears the structural bar that ClariQ and several reconstructed
environments did not:

- hidden context is released and coherent;
- observations are generated conditionally on full dialogue history;
- most useful checklist information arrives after the first turn;
- next actions measurably track newly revealed answer content;
- trajectories differ materially across repeated runs;
- the endpoint remains unsaturated for a meaningful fraction of cases.

The next authorized step is a small, separately preregistered mechanics gate on
disclosed IDs only. It must test the first causal link directly: whether a
dynamic non-myopic score ranks first probes by realized future checklist
discovery better than immediate value, fixed support, answer-shuffled, and
random controls.

V2 development and holdout remain unread. OpenRouter calls/cost: `0 / $0`.
OatML jobs: `0`.
