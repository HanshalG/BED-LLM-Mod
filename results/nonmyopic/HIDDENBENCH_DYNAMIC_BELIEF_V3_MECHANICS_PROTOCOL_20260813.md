# HiddenBench Dynamic-Belief V3 Mechanics Protocol

Date frozen: 2026-08-13

Status: **prospective final reserve mechanics. No V3 task semantic value, model
response, registered answer, planner score, or endpoint has been opened.**

## Scientific Question

Does depth-two design outperform compute-matched one-step EIG because it values
the quality of the LLM's own response-conditioned regenerated belief state?

This preserves V2's scientific interface and all numerical gates. It changes only
the cohort and strengthens endpoint process isolation after V2's pre-serving
ordering failure.

## Binding and Cohort

- HiddenBench commit:
  `3be6ca16973e4fb751ffc0dfb7eb11f2d28335d1`;
- benchmark SHA-256:
  `2815afffca4e470d1dfbc81e625160447df1109ce371968181c9e1e6b90443a3`;
- original source protocol / manifest / audit SHA-256:
  `5340d17a11ae4654a82fff76a7e4244c39336f3b282240c05c6d0b2c88e5be28`,
  `b105c1f54e2b5ec56606b4eef2c7f464e2bdab996315f9a54a4d9ea817b6d56d`,
  and `67b3f641bb8ac7956e33281c2bdd074402e775ac88c643f6db4c468788b73bf9`;
- V1 terminal report SHA-256:
  `ac03b68f4fc367fc6d30ff7d53911eeb7d0ea1c3207083bac1df11a7d8970349`;
- V2 ordering terminal report SHA-256:
  `7ab19c902d432e9721e9f3cde2ec20f39064979d7c4f5607449157dd92d3a224`;
- V3 cohort: positions 4--7 of the original nine-row reserve, ordered-ID
  SHA-256
  `822314c2c662a2711b6d2253601b9a801071c4d53bb53b92bb115240d98bee40`;
- this excludes all V1 and V2 rows and leaves reserve position 8 unused.

No replacement, semantic filtering, or content-dependent selection is permitted.
The public cohort manifest contains only bindings, counts, and hashes.

## Exact Ten Calls

- exact model: `deepseek/deepseek-v4-flash-0731`;
- reasoning explicitly disabled and excluded;
- temperature `0`;
- seeds `202608135000`--`202608135009` in request order;
- exact structured JSON required;
- maximum output tokens 8,192;
- concurrency at most 4;
- retries 0, positive configured backoff `1.0` seconds;
- forced-final continuations disabled;
- expected accepted requests / maximum HTTP attempts 10 / 10;
- per-request reservation `$0.004`, run cap `$0.040`.

Calls are four root calls, four answer-conditioned refresh calls, one joint
private-fact router, and one independent joint semantic auditor. Root calls return
a prior, four semantic evidence requests, three semantic response channels per
request, and every `P(channel | option, query)`. Refresh calls independently
regenerate one normalized option belief for every one of the 12 `(query, channel)`
branches. Router and auditor independently map all 16 queries to one private fact
ID and one response channel ID. Local code admits only exact released facts from
an agreed mapping.

## Policies and Scores

Dynamic depth two values each first query by root response probabilities, then for
every branch uses the LLM-regenerated belief to choose the best unused second
query and integrates the exact second-step likelihood entropy. Compute-matched
myopic uses the exact same ten responses, likelihoods, regenerated beliefs, width,
and realized updater, but chooses the first query using one-step EIG only.
Fixed-support depth two replaces regenerated beliefs with exact Bayes posteriors.
Random uses local seed `202608135100` and no model call. Lexical query ID breaks
all ties.

## Label-Free Gates

All are conjunctive and unchanged from V2.

1. Exact 10 accepted/HTTP requests, zero retries/reasoning/forced exits/finals,
   complete nonempty parse, and cost at most `$0.040`.
2. Exact schemas, IDs, complete option/query/channel/branch coverage, finite
   probabilities, normalization within `1e-6`.
3. Four distinct requests and dimensions per task; no request contains `correct
   answer`, `which option`, or `choose the answer`.
4. Three distinct channels/query; at least two queries/task have max option-row
   TV at least `0.10`.
5. Every task has max root EIG at least `0.005` nats, EIG range at least `0.001`,
   and myopic winning margin at least `0.0001`.
6. At least 10/12 regenerated branches/task increase received-channel
   compatibility versus root prior; mean increase at least `0.01`.
7. Per task mean regenerated-vs-exact-Bayes TV in `[0.01,0.20]`, at least 6/12
   branches with TV at least `0.01`, and mean within-query pairwise regenerated
   TV at least `0.10`.
8. Dynamic depth two winning margin at least `0.0001` on every task, first-query
   disagreement with matched myopic on at least 2/4 tasks, and at least two
   response-contingent first-query branches per task.
9. Router/auditor agreement on exact private-fact and channel IDs for all 16
   queries, at least three distinct facts used per task, and no generated
   observation text admitted.
10. Prompt replay proves no source/task ID, correct-answer designation, rationale,
    or private fact in root/refresh prompts, and no option in router/auditor prompt
    structure.

An independent verifier must reconstruct schemas, parses, metrics, policies, and
all ten gates without importing the producer's parser or scorer.

## Mandatory Endpoint Isolation

V3 uses two physically separate custodians:

- the serving custodian has no endpoint function or mode and emits only
  planner/router views;
- the endpoint custodian has no planner/router function or mode.

The endpoint custodian requires a JSON pass token and independently verifies all
of these before reading the benchmark:

1. exact V3 execution binding SHA-256;
2. exact V3 cohort manifest and source-audit SHA-256;
3. exact complete raw-response-bank SHA-256;
4. exact label-free result and independent verification SHA-256;
5. result status `serving_pass`, authority `endpoint_only`, and all label-free
   gates true;
6. verification status `verification_pass` and all replay gates true;
7. zero endpoint files already exist.

Only then does it read the benchmark and emit four `{slot, correct_option_id}`
objects to a private endpoint file. The source, serving, and verifier tests may
never import or execute the endpoint custodian against the real source. Endpoint
logic is rehearsed only on synthetic rows; a static test rejects real-source
endpoint invocation strings outside the dated execution wrapper.

## Endpoint Gates

Every policy executes two distinct queries using the shared consensus channel
map. Dynamic, matched myopic, and random use the regenerated belief after step 1
and exact likelihood updating after step 2; fixed-support uses exact Bayes twice.
Multiclass Brier is `sum_option (p-option_one_hot)^2`, log loss is
`-log p(correct)`, and correct probability is also reported.

Development is authorized only if:

1. dynamic and matched myopic first queries differ on at least 2/4 tasks;
2. dynamic Brier is strictly lower on at least 3/4 and mean paired improvement is
   at least `0.01`;
3. dynamic mean log loss is strictly lower than matched myopic;
4. dynamic is nonworse than fixed-support depth two in mean Brier and log loss
   within `1e-9`;
5. dynamic is strictly better than random in mean Brier and log loss;
6. every policy has four finite nonsaturated rows.

A pass authorizes only a separately frozen development protocol on the original
16-task development split. It does not authorize confirmation or a headline.

## Rehearsal and Stop Rules

Before binding, zero-call tests must instantiate the real adapter and construct
all ten exact payloads, proving positive backoff, zero retries, disabled reasoning,
seeds, schemas, token caps, and reservations. They must run the entire producer,
independent verifier, pass-token verifier, and endpoint scorer on synthetic tasks.
No test may invoke the endpoint custodian on real source.

Any source, privacy, rehearsal, transport, schema, semantic, replay, endpoint, or
budget failure closes this exact V3 cohort/interface without retry, repair,
replacement, threshold relaxation, or endpoint-informed successor. Reserve
position 8 is never a same-protocol replacement.
