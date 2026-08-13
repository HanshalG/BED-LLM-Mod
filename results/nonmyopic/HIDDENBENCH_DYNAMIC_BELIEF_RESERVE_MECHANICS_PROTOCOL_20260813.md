# HiddenBench Dynamic-Belief Reserve Mechanics Protocol

Date frozen: 2026-08-13

Status: **prospective, scientifically distinct reserve-cohort mechanics. No
reserve task semantic value has been inspected, serialized, or used to choose
this interface; no model response, registered answer, planner score, or endpoint
has been opened under this protocol.**

## Question

Can depth-two Bayesian experimental design improve a semantic decision because it
values the quality of the language model's own answer-conditioned belief state,
which one-step EIG cannot see?

This is not a rerun or repair of the terminal semantic-query V1. V1's four
mechanics rows, prompts, seeds, and interface remain closed. V2 uses four untouched
rows from the previously frozen reserve and a different scientific object:
answer-conditioned **regenerated beliefs**, not merely a fixed likelihood table.

## Immutable Source and Cohort

- HiddenBench commit:
  `3be6ca16973e4fb751ffc0dfb7eb11f2d28335d1`;
- benchmark SHA-256:
  `2815afffca4e470d1dfbc81e625160447df1109ce371968181c9e1e6b90443a3`;
- source protocol SHA-256:
  `5340d17a11ae4654a82fff76a7e4244c39336f3b282240c05c6d0b2c88e5be28`;
- source manifest SHA-256:
  `b105c1f54e2b5ec56606b4eef2c7f464e2bdab996315f9a54a4d9ea817b6d56d`;
- machine source audit SHA-256:
  `67b3f641bb8ac7956e33281c2bdd074402e775ac88c643f6db4c468788b73bf9`;
- terminal V1 report SHA-256:
  `ac03b68f4fc367fc6d30ff7d53911eeb7d0ea1c3207083bac1df11a7d8970349`;
- V2 cohort: positions 0--3 of the exact nine-row reserve in the frozen source
  hash order. It is disjoint from V1 mechanics, opportunity, development, and
  confirmation. No replacement, filtering, or content-dependent selection is
  permitted. Its ordered opaque-ID SHA-256 is
  `adcbabc78acc15d7b88d2b0552636525ea64d25840113033a469c558bae3807a`.

The public V2 manifest may contain bindings, counts, and ordered-ID hashes only.
It may not contain a task ID, name, description, fact, answer, rationale, source
row, model response, or endpoint.

## Privacy and Ordering

A short-lived custodian projects the four selected rows into disjoint views:

- **planner**: opaque slot, description, shared facts, and opaque option IDs plus
  option strings;
- **router**: opaque slot, description, indexed private facts, and no options;
- **endpoint**: opaque slot and the correct option ID only.

The `correct_answer` designation and rationale are absent from every planner,
router, auditor, and model process. The endpoint view is not created until all ten
responses are banked, strict serving replay passes, and every label-free serving
gate passes. A serving failure opens no endpoint view.

Raw selected values and model responses remain under gitignored `private/` paths.
Public results contain only opaque IDs, aggregates, scores, gates, accounting, and
dispositions.

## Exact Ten-Call Interface

- model: exact `deepseek/deepseek-v4-flash-0731`;
- reasoning: explicitly disabled and excluded;
- temperature: `0`;
- seeds: `202608134000` through `202608134009` in request order;
- structured JSON is mandatory;
- maximum output tokens: 8,192;
- concurrency: at most 4;
- retries: 0;
- backoff configuration: strictly positive `1.0` seconds even though retries are
  zero;
- forced-final continuations: disabled;
- expected accepted requests / maximum HTTP attempts: 10 / 10;
- per-request reservation: `$0.004`;
- run cap: `$0.040`.

The exact call order is:

1. four root calls, one per task;
2. four answer-conditioned refresh calls, one per task;
3. one joint private-fact router call over all four tasks;
4. one joint independent semantic-auditor call over the same four tasks, without
   seeing the router response.

Each root call returns:

- one normalized prior over all three or four opaque answer options;
- exactly four distinct semantic evidence requests `Q1`--`Q4`;
- exactly three shared semantic response channels `C1`--`C3` per query;
- a normalized `P(channel | option, query)` row for every option and query.

Each refresh call receives the public task and the validated root object. For all
12 `(query, channel)` branches it independently returns one normalized regenerated
belief over the same opaque options. It is instructed to reason from the semantic
meaning of the hypothetical response, not merely copy the numeric Bayes update.

The router and auditor each receive the four router views plus all validated query
and channel descriptions. Independently, each maps all 16 queries to exactly one
private-fact ID and one response-channel ID. Local code admits only the exact
released fact selected by a consensus mapping; no generated observation text can
enter a history.

## Exact Policies

Let `pi` be the root prior and `L_q(c|theta)` the root likelihood. Let
`R(q,c)` be the LLM-regenerated belief after hypothetical response channel `c` to
query `q`.

### Dynamic depth two

For each first query `q`, weight branches by
`p(c|q) = sum_theta pi(theta)L_q(c|theta)`. Within each branch, select the unused
second query that maximizes one-step EIG from `R(q,c)` under its root likelihood.
The first-query score is root entropy minus expected terminal entropy after that
second observation. Ties use lexical query ID.

### Compute-matched myopic

Use the exact same ten responses and full root/refresh bank. Select the first query
using only one-step EIG from `pi`; after the realized first response, use the same
`R(q,c)` updater and choose the second query greedily. Thus model calls, candidate
width, likelihoods, regenerated beliefs, and execution updater are identical; only
the first-action objective differs.

### Fixed-support depth two

Replace every `R(q,c)` with the exact Bayes posterior implied by `pi` and `L` and
otherwise exhaustively apply the same depth-two scorer. This is the classical
fixed-support control.

### Random

Choose a first query using local seed `202608134100`, then use the realized
`R(q,c)` and greedily choose the best unused second query. It makes no model call.

All policies use the same consensus router outcomes for common-random-number
execution.

## Label-Free Serving Gates

All gates are conjunctive.

1. Exact ten accepted requests and ten HTTP attempts; zero retries, reasoning
   tokens, forced exits/finals, empty responses, or parse failures; cost at most
   `$0.040`.
2. Exact keys, IDs, option coverage, branch coverage, finite probabilities, and
   normalization within `1e-6` for every object.
3. Every task has four pairwise-distinct requests and target dimensions. Direct
   answer requests containing `correct answer`, `which option`, or `choose the
   answer` are invalid.
4. Every query has three distinct channel descriptions. At least two queries per
   task have maximum option-row total-variation distance at least `0.10`.
5. For every task, maximum root one-step EIG is at least `0.005` nats, its range is
   at least `0.001`, and the winning myopic score margin is at least `0.0001`.
6. Refresh answer obedience: per task, at least 10/12 regenerated branches increase
   expected compatibility with the received channel relative to the root prior;
   the mean compatibility increase is at least `0.01`.
7. Refresh calibration and irreducibility: per task, mean total-variation distance
   from exact Bayes is in `[0.01, 0.20]`, at least 6/12 branches differ by at least
   `0.01`, and mean within-query pairwise TV among regenerated channel beliefs is
   at least `0.10`.
8. Dynamic depth two has a winning score margin at least `0.0001` on every task,
   changes the first query relative to compute-matched myopic on at least two of
   four tasks, and has response-contingent second queries on at least two first
   queries per task.
9. Router/auditor semantic obedience: exact private-fact agreement on all 16
   queries, channel agreement on at least 14/16, at least three distinct private
   facts used per task, and no direct-answer or free-text observation admitted.
10. Prompt privacy replay proves no task/source ID, `correct_answer` designation,
    rationale, or private fact enters a root/refresh prompt, and no option enters a
    router/auditor prompt.

The independent verifier reparses raw responses and reconstructs all metrics
without importing the producer's parser or scorer. Any disagreement fails closed.

## Endpoint and Opportunity Gates

Only after all label-free gates and independent replay pass may the endpoint
custodian map each sealed correct answer string to its opaque option ID.

Each policy executes exactly two distinct queries using the shared consensus
channel table. Dynamic, compute-matched myopic, and random apply the regenerated
belief after the first channel and then exact likelihood updating after the second;
fixed-support depth two applies exact Bayes at both steps.

Report per-task and mean final Brier score, correct-option log loss, and correct
option probability. Development is authorized only if all are true:

1. dynamic and compute-matched myopic choose different first queries on at least
   two tasks;
2. dynamic has strictly lower Brier than compute-matched myopic on at least three
   of four tasks and mean paired Brier improvement at least `0.01`;
3. dynamic mean correct-option log loss is strictly lower than compute-matched
   myopic;
4. dynamic is nonworse than fixed-support depth two in both mean Brier and mean log
   loss within tolerance `1e-9`;
5. dynamic is strictly better than random in mean Brier and mean log loss;
6. every policy has four finite, nonsaturated endpoint rows.

A pass authorizes only a separately frozen development protocol on the original
16-task development split. It does not authorize confirmation or a paper
headline.

## Mandatory Rehearsal and Stop Rules

Before an execution binding can be committed, synthetic tests must instantiate
the real adapter, construct all ten exact payloads, and assert positive backoff,
zero retries, disabled reasoning, seeds, schemas, token caps, and cost reservations.
No network request is made in rehearsal.

Any source, rehearsal, transport, schema, semantic, replay, or endpoint failure
closes this exact V2 cohort, prompts, model, seeds, and interface. There is no
retry, repair, task replacement, threshold relaxation, or endpoint-informed
successor. A failure is reported at the stage actually reached and cannot be
described as planner or policy evidence if its predecessors did not pass.
