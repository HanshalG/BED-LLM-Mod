# Prospective native hypothesis-revision qualification

## Scientific question

Does revealing a second real observation to the LLM proposer produce transferable
new hypotheses beyond compute-matched history-blind generation and exact
conditioning of the original support? This is a belief-update prerequisite,
not a non-myopic planning or monotonic-depth claim.

## Fixed intervention and controls

Use four fresh source-qualified tasks. Each has one initial demonstration,
one predetermined second observation, two future query inputs and seven target
inputs. All input grids are public. Only the initial output is visible initially.

Generate the shared initial pool through three Luna-medium calls: four mechanism
plans, eight Python implementations, one eight-program repair. Bank all16attempted
slots before revealing the prescribed second output once. Do not select the
second example based on forecasts or endpoint labels.

Each revision arm uses three calls of the same form and receives the same initial
plans/16slots. The aware proposer sees both observations. The blind proposer sees
only the first; it receives neither second-answer execution feedback nor any
aware-arm output. Both get the same public inputs and runtime contract. Alternate
arm order by task index. Each final pool is the original16 plus its own16new slots.
Plans, compile calls and repair calls are separately banked. No retries, refills,
extra repairs, partial-batch salvage or reasoning changes.

Both final pools are numerically conditioned on BOTH observed examples using
the identical unpruned fixed-slot scorer. Thus the intervention changes proposal
generation, not the information available to the posterior filter. Also score
the original16slots conditioned on both examples as a zero-call reference; do
not confuse this weaker no-generation baseline with the matched blind control.
Future failures retain probability mass. Empty support predicts unit failure.

## Source and endpoint rules

Reserve all24candidates from the previous source-qualified study, including the
17unattempted candidates; do not reuse them. Exclude their union with the prior
50closed IDs,74in total. Freeze a new24candidate list using first24 ascending
SHA256(`bed-rearc-native-revision-v1:` + ID). Use the established bounded source
screen: only verify-phase ValueError may reject a candidate and advance; all
other failures abort. Stop at four tasks whose exact11channels verify, maximum
264source attempts. Inference is restricted to that reference-valid population.

Seeds: initial50100, second50200, future queries50201/50202, targets50300..50306.
The source journal initially exposes only the50100output; all ten other outputs
are hashes. For each task,50200may open only after its initial pool is banked.
The remaining36outputs may open once only after all four tasks' three forecasts
are sealed and the observed-coverage gate passes. Never pass task IDs, source
functions or withheld reference outputs into the proposer or candidate runtime.

## Frozen gates

Require aware two-example-consistent support on at least3/4tasks before opening
the36future labels. If this fails, bank coverage null without future outputs.
After sealing and permitted opening, require every following condition:

- Positive aware probability on at least6/8realized future query answers.
- At least2/4tasks with distinct positive-mass, nonfailure predictions on a
  future query input.
- Aware mean whole-grid Brier improves on blind by at least0.01 over36outputs,
  with task-level improvements of at least0.01 on at least2/4tasks.
- Aware fixed-canvas Brier is no worse than blind.
- On at least2/4tasks, aware support covers a future query answer that the
  initial pool conditioned on both examples does not cover.

Report all per-task/query/target scores and support, empty pools, literal-output
flags as descriptive diagnostics, actual cost/tokens and updated plan mechanisms.
Do not count slots or cells as independent experimental units. No threshold
changes after responses. A pass opens only a separate prospective simulation
fidelity and horizon-opportunity test; it never sets depth_authorized=true here.
The four-task size is an inexpensive falsification screen, not a powered
population-level positive claim.

## Calls and cost

Nine calls per task,36maximum. Exact openai/gpt-5.6-luna medium with existing
named-plan schema and bounded transport contract; OpenAI-only/no fallback.
Seeds50400+6*i+[0,1,2] for initial stages and +[3,4,5] for both arms' corresponding
revision stages. Matching seeds do not guarantee deterministic model responses.
Reserve .08 per HTTP attempt, full2.88block before launch, under the hard
account-wide London-day5ceiling. Recheck actual catalog and cumulative usage
before dispatch. No other paid experiment is authorized by this protocol.

The source collector, four-task scoring panel, reveal/endpoint bank, budget
wrapper and exact failure replay must be implemented and tested before calls.
Current native_revision controller tests are not a substitute for that readiness.
Freeze candidate manifest and all implementation bindings before their own
responses. Closed previous cohorts and failed gates remain untouched.
