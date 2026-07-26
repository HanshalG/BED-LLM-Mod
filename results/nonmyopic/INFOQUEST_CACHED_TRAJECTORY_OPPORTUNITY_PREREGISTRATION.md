# InfoQuest Cached-Trajectory Opportunity Preregistration

Frozen after calibrating mechanics on the disclosed IDs `{0, 1, 4}` and before
loading any trajectory from the 80-ID opportunity split.

## Question

Do the released InfoQuest dialogues exhibit prevalent delayed information
discovery and observation-conditioned next-turn behavior, with enough
cross-run variation to justify a new LLM-native ranking-fidelity experiment?

This audit is structural. It cannot establish causal policy efficacy, compare
planning methods, or support a non-myopic performance claim. Each released
trajectory contains one realized future and therefore has hindsight that a
deployed policy would not have.

## Frozen Sources

- Source manifest interface: `infoquest-llm-bed-manifest-1`
- Opportunity IDs: exact frozen 80-ID split, SHA-256
  `737296c449680cfb7c78aa03d44508af487eb9624d0e416d37cf5ddbaa01958b`
- Hidden settings: pinned source hash
  `f8b1a8a0e692ce9877bb216794b6c8574a04794770a4c0b924f91dfddcbc08bb`
- Three official Falcon3-7B, Gemini-2.0-Flash-user-simulator, Selene-judge
  30-turn baseline runs, SHA-256:
  - `454039500859fd9a36146908a2a2137cfcf6b358d89826c7cffea8445d07f0c1`
  - `518a39ec19dd3b6cdfbdf2fec81c8d1f6d6811091ff18b873d335628d39eb9cb`
  - `296c07e45afa87bc3dafe559b62f1c836ae1903a736c81410602e9ebf5197d15`

ID `4` is quarantined under the trajectory-access amendment and is absent from
opportunity. Every baseline file must contain exactly one record for every ID
`0..499`; records are selected through an explicit ID map, never row position.
Development and effective holdout remain unread.

## Frozen Validation

For each of `80 IDs x 2 hidden settings x 3 runs = 480 trajectories`:

1. require the exact released record schema;
2. require a system message followed by exact alternating policy (`user`) and
   simulator-observation (`assistant`) messages, ending in a policy message;
3. require 2--30 policy turns and one evaluation per policy turn;
4. require integer checklist reward in `0..5`, monotone across turns;
5. require `done` exactly when final reward is five;
6. require any incomplete trajectory to use all 30 turns.

Any source, schema, ID, alternation, alignment, or reward violation fails the
audit without repair.

## Frozen Metrics

Delayed discovery:

- fraction with at least two turns;
- fraction with at least two checklist items unresolved after the first turn;
- fraction gaining at least two checklist items after the first turn;
- mean delayed gain, initial reward, final reward, completion, and turns.

Observation-conditioned uptake:

1. lowercase ASCII-alphanumeric tokenization;
2. remove the fixed stopword set and tokens shorter than three characters;
3. at transition `t`, define novel observation tokens as tokens in the
   simulator answer not previously seen in the seed message, policy messages,
   or earlier observations;
4. count novel tokens reused in the immediately following policy message;
5. compare against the same next policy message using the next simulator
   observation in a circular one-step shift, with the same prior-token set.

Cross-run variation, for each of 160 task/world cells:

- fraction whose turn-count range across three runs is at least two;
- fraction with three distinct exact first-policy-message hashes;
- fraction whose final checklist reward varies.

The audit emits only IDs, numeric metrics, reward traces, and message/evaluation
hashes. It emits no seed, setting, message, trait, constraint, solution, or
checklist text.

## Conjunctive Gates

Structural gates above must all pass, plus:

| Metric | Threshold |
| --- | ---: |
| Multi-turn fraction | at least `.95` |
| At least two initially unresolved | at least `.80` |
| Delayed gain at least two | at least `.75` |
| Mean delayed gain | at least `2.0` |
| Immediate-minus-shifted novel uptake per transition | at least `.25` |
| Trajectories with positive uptake advantage | at least `.60` |
| Task/world cells with turn-count range at least two | at least `.40` |

All conditions are conjunctive. A pass establishes only that InfoQuest has a
prevalent path-dependent sequential opportunity. It authorizes a separately
preregistered, paid mechanics ranking gate on disclosed IDs only. That gate
must compare a dynamic non-myopic score against immediate-value, fixed-support,
answer-shuffled, and random controls before any opportunity policy experiment.

A failure closes this exact cached-trajectory route without changing
tokenization, shift control, thresholds, baseline family, or split.

OpenRouter calls/cost: `0 / $0`. OatML jobs: `0`.
