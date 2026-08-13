# HiddenBench Semantic-Query Serving Protocol

Date frozen: 2026-08-13

Status: **prospective ten-call serving gate. No mechanics task content, private
fact, model response, planner score, or registered answer has been opened under
this protocol.**

## Scientific Question

Can one fixed nonreasoning language-model interface produce the semantic objects
needed for genuinely adaptive depth-two Bayesian experimental design on
HiddenBench?

The action is a natural-language request for one kind of private evidence. It is
not a participant index. This distinction is essential: HiddenBench shuffles
private facts among otherwise exchangeable participants, so choosing Person 1
before Person 2 cannot itself establish a non-myopic opportunity. A broad first
request can, however, reveal which targeted second request is useful.

The load-bearing LLM objects are:

1. four candidate evidence requests and a prior over the released answer
   options;
2. for every request, a shared response-channel support, option-conditioned
   likelihoods over that support, and one follow-up query ID per response
   channel;
3. a private-fact router that maps a realized request to exactly one released
   private fact by index.

The release supplies none of the option-conditioned response worlds or
likelihoods in item 2. They must be generated semantically. The router in item 3
cannot emit free text: local code returns the exact released fact selected by the
model, preventing a hallucinated observation from entering the history.

## Immutable Source Binding

- source protocol SHA-256:
  `5340d17a11ae4654a82fff76a7e4244c39336f3b282240c05c6d0b2c88e5be28`;
- source manifest SHA-256:
  `b105c1f54e2b5ec56606b4eef2c7f464e2bdab996315f9a54a4d9ea817b6d56d`;
- source result SHA-256:
  `67b3f641bb8ac7956e33281c2bdd074402e775ac88c643f6db4c468788b73bf9`;
- HiddenBench commit:
  `3be6ca16973e4fb751ffc0dfb7eb11f2d28335d1`;
- benchmark SHA-256:
  `2815afffca4e470d1dfbc81e625160447df1109ce371968181c9e1e6b90443a3`;
- mechanics split: the exact four rows at positions 0--3 of the already frozen
  source hash order. No replacement, filtering, or content-dependent ordering is
  permitted.

## Privacy Boundary

A short-lived data-custodian process may parse the bound source and project the
four selected rows into two disjoint in-memory views:

- the **planner view** contains an opaque slot, description, shared facts, and
  option strings;
- the **router view** contains the same opaque slot, description, and indexed
  private facts, but no answer options.

The `correct_answer` field, its designation of one option as correct, and the
rationale are never emitted by the custodian, written to an intermediate
artifact, passed to an adapter, or loaded by the planner/router process. The
planner necessarily sees all candidate option strings, including the string that
the sealed field later designates as correct, but it receives no signal about
which option that is. The custodian exits before a model call. Public outputs
contain only aggregate counts, hashes, gates, accounting, and dispositions; raw
task values and model responses are banked under a gitignored `private/`
directory.

The official answer may be opened only by a separately frozen endpoint evaluator
after serving and opportunity mechanics pass. A serving failure opens no answer.

## Exact Model Interface

- model: `deepseek/deepseek-v4-flash-0731`;
- reasoning: explicitly disabled and excluded;
- temperature: `0`;
- provider routing: structured JSON required, no fallback that drops the schema;
- seeds: `202608133000` through `202608133009`, assigned in request order;
- maximum output tokens: 4,096;
- concurrency: at most 10;
- expected accepted requests: 10;
- maximum HTTP attempts: 10;
- retries and forced-final continuations: 0;
- per-request reservation: `$0.0015`;
- serving run cap: `$0.015`.

Immediately before dispatch, the authenticated catalog must still report the
exact model and non-increased input/output prices relative to `$0.09/$0.18` per
million tokens. A price increase, malformed catalog response, changed model ID,
or insufficient account-wide Aug-13 allowance fails before any request.

## Exact Ten Calls

Calls are made in this order, although they may execute concurrently within each
block:

1. four root calls, one per mechanics task, each returning exactly four unique
   semantic evidence requests (`Q1`--`Q4`) and one normalized prior over every
   opaque option ID;
2. four world-model calls, one per task, each returning a model for every root
   request. Each query model has exactly three common response channels
   (`C1`--`C3`), one normalized channel distribution for every option, and one
   follow-up query ID for every channel. A follow-up must select one of the other
   three already modeled root requests;
3. two router calls for `Q1` and `Q2` of mechanics position 0. Each receives only
   the router view and returns `addressed`, a valid private-fact ID, and no
   observation text. Local code substitutes the exact selected fact.

No call may see the `correct_answer` field or designation, rationale, task ID,
source split, or a private fact outside the two router prompts. The root/world
calls see all opaque candidate options but no private facts. The router calls see
no answer options or option-conditioned worlds.

## Frozen Serving Gates

All gates are conjunctive.

### Transport and schema

1. Exactly ten accepted requests and ten HTTP attempts, with zero retries,
   reasoning tokens, forced-final requests, empty responses, or unparsed rows.
2. Every root, world, and router object has exactly the frozen keys, IDs, lengths,
   finite values, and coverage. Priors and likelihood rows are nonnegative and
   sum to one within `1e-6`.
3. No request contains the `correct_answer` field/designation or rationale, and
   no root/world request contains a private fact. No response or public artifact
   contains a source task ID.

### Semantic mechanics

4. Every task has four pairwise-distinct request strings and target dimensions.
   A request containing direct-answer language such as “correct answer”, “which
   option”, or “choose the answer” is invalid.
5. Every query has three pairwise-distinct channel descriptions. For at least two
   queries per task, option-conditioned likelihood rows differ by total variation
   distance at least `0.10`.
6. For at least two queries per task, at least two channel-conditioned follow-up
   query IDs differ. No follow-up may equal its parent query. Independently of
   the model's suggested IDs, the evaluator exhaustively scores every remaining
   root request after every first-response channel; generated suggestions are a
   semantic-branch diagnostic, not the optimizer.
7. The generated finite model has positive nondegenerate one-step information
   gain and a well-defined exhaustive depth-two policy on all four tasks. For a
   first query `q`, depth-two value is its expected entropy reduction plus the
   expected maximum second-step entropy reduction over `Q1`--`Q4` excluding `q`,
   under the posterior for each first-response channel. Depth two must select a
   different first request from greedy one-step EIG on at least two tasks. On
   every task, maximum one-step EIG must be at least `0.005` nats, the one-step
   EIG range must be at least `0.001` nats, and both the greedy and depth-two
   winning scores must exceed their runners-up by at least `0.0001` nats. This is
   a serving-mechanics diagnostic, not an efficacy endpoint.
8. Both router calls report `addressed=true`, return valid fact IDs, and select
   different private facts for the two distinct requests. The realized
   observations are byte-for-byte released facts selected by those IDs; no model
   text is admitted as evidence.

## Interpretation and Stop Rules

A pass establishes only that the semantic action, likelihood, branch, and exact
observation interfaces are executable and nondegenerate. It authorizes freezing
a four-task opportunity-mechanics protocol. That descendant must still use
common-random-number realized responses and compare depth two against:

- greedy one-step EIG;
- a compute-matched answer-free myopic ensemble using the same number of world
  generations;
- fixed-root and random-query controls.

Before any development call, depth two must change the first request and improve
registered-answer log probability or Brier score on at least three of the four
mechanics tasks. Dynamic branch worlds must also differ materially from the
answer-free control.

A failure at any serving gate closes this exact interface, seeds, and mechanics
cohort. It cannot be repaired by retrying, replacing tasks, weakening a threshold,
changing prompts, or opening answers. No serving result supports a policy-efficacy
or paper headline.
