# LLM-Native Benchmark Source Audit

Audited: 2026-07-25. This is a zero-call source audit. It does not use benchmark
test outcomes as model evidence and it authorizes no paid run.

## Sources

| Benchmark | Source | Pinned revision | Local source state |
| --- | --- | --- | --- |
| RegretBench | paper: `https://arxiv.org/abs/2607.21143`; cited repository: `https://github.com/ngocminhta/RegretBench` | arXiv source downloaded 2026-07-25 | Paper available; cited repository returns 404 and no dataset was found on Hugging Face |
| EComAgentBench | `https://github.com/Morizeyao/EComAgentBench_` | `867dcc59957d4c5f89e9cfd701ee20f91ef96a83` | Code and 662 benchmark rows available |
| pi-Bench | paper: `https://arxiv.org/abs/2605.14678`; code: `https://github.com/Simplified-Reasoning/Pi-Bench` | `383910b1698758a198b86037c63a111c8edc32ad` | Code, 100 tasks, five episodes, and task assets available |
| CRA-Bench | `https://huggingface.co/datasets/l1i1p/CRA-Bench` | `30d6c7c7e67e7dd21904c1aae67aa6fc8d596640` | 750 task rows and exact labels available; runner, simulator, catalog, and paper link absent |
| DiscoverLLM | code: `https://github.com/tsook/discoverllm`; data: `https://huggingface.co/datasets/kixlab/DiscoverLLM-multiturn-preferences` | code `a9eb2846`; data `c857bbf6` | Simulator and 9,318 candidate rows available; candidate counterfactual states/trajectories omitted |
| IG-clarifier | paper: `https://arxiv.org/abs/2606.03135`; code: `https://github.com/Demi-deng2/IG-clarifier` | code `4071af11` | Code-only single-turn reward interface; task data, tau-Bench harness, trajectories, simulator service clients, and compatible `verl` checkout absent |
| OPEN | paper: `https://arxiv.org/abs/2403.05534` | arXiv v1 | Greedy one-step BOED over a fixed Bradley--Terry feature model; promised code and anonymized user data were not linked from the paper |
| BIRD-Interact | code: `https://github.com/bird-bench/BIRD-Interact`; data: `https://huggingface.co/datasets/birdsql/mini-interact` | code `451fe2c3`; data `10253f23` | 300 SQLite tasks and a clarification simulator; one annotated interpretation per task term, with no released latent-world prior |

## RegretBench

RegretBench is the closest conceptual match because its paper defines
Conversational Information-Gathering environments with latent intents, semantic
actions, observations, and a reference planner. The public paper source contains
the manuscript but not the generated environments, transition tables, or
reference-planner implementation. The cited GitHub repository still returns 404,
GitHub search does not expose another official repository, and Hugging Face
search does not expose the benchmark data.

**Decision:** keep RegretBench first in the retry queue, but do not reconstruct
or label an unofficial approximation as a RegretBench result. A source release
would immediately justify a new structural audit before model calls.

**2026-07-26 recheck:** arXiv v1 now exposes the full benchmark formulation and
reports 4,836 AmbigDocs, 1,451 CondAmbigQA-2K, and 377 PSCon CIGs. It confirms
that the desired endpoint is exact hidden-intent success plus interaction cost
and regret against a semantic reference planner. The generated CIGs, semantic
question mapper, user simulator, and planner are still absent from the cited
404 GitHub repository and from Hugging Face search. Reconstructing them from the
three source datasets would invent the benchmark transition model and reference
policy. The retry-queue decision therefore remains unchanged.

## EComAgentBench

The released benchmark has 662 product-recommendation tasks. Its hidden
clarification content is reproducible and externally specified, which is useful:

- 660 tasks have exactly two clarification slots and two have three;
- every slot links to exactly one clarification rubric;
- all tasks allow ten clarification turns;
- `ask_user` reveals the single unrevealed slot with the highest keyword match;
- slots do not condition on earlier answers and do not create new actions or
  state-dependent follow-ups.

The task therefore has semantic hidden information but no native clarification
planning bottleneck. An agent can ask for all two or three slots within the
ten-turn allowance, and slot order does not affect attainable clarification
coverage. A sequential story would have to combine clarification with product
search over the separately downloaded product database; it would not arise from
the released clarification graph itself.

**Decision:** do not download the approximately 25 GB product database or run a
paid agent. Revisit only if a zero-cost clarify-then-search construction first
shows a strict lower-immediate/higher-terminal root on a frozen cohort.

## pi-Bench

pi-Bench is the strongest current source for a future LLM-native result. The
paper and release provide natural underspecified requests, persistent workspaces,
cross-session dependencies, hidden semantic requirements, targeted
clarification, and artifact endpoints. The source inventory is:

- 100 tasks across five personas and 524 hidden intents;
- 30 dependency-final tasks and 45 dependency edges, exactly nine edges per
  persona;
- 225 hidden intents in dependency-final tasks, with 3--20 intents per task
  (median 6.5);
- 27 tasks with deterministic tool-evaluation scripts;
- 510 textual checklist criteria, evaluated by an LLM unless supplemented by a
  task-specific tool script.

The dependency mechanism is real. Sessions share a persistent workspace, the
paper's history ablation reports lower proactivity without prerequisite
sessions, and many final-task intents explicitly inherit prior conventions or
facts. This is precisely the kind of semantic memory recovery where an LLM is
hard to replace with a fixed enumerator.

### Why the release is not yet a non-myopic BED environment

1. Every listed hidden intent is active. The benchmark does not release
   mutually exclusive intent hypotheses or a prior over alternative user
   worlds.
2. The user simulator uses GPT-5.4 at temperature zero to judge whether the
   response already satisfies each intent and whether a question targets it.
   This is an LLM-mediated transition, not an exact likelihood table.
3. If no targeted question matches, the simulator reveals the first unmet
   intent automatically. If several intents match, one response can reveal all
   of them.
4. All intents are eventually revealed and the session limit is 30 turns, while
   the largest task has 20 intents. There is no native information-acquisition
   scarcity or terminal penalty for leaving an intent unknown.
5. `depends_on` determines dependency groups for aggregate weighting, while
   task execution follows episode order with a persistent workspace. The
   release does not provide a reference planner or action-conditioned dependency
   transition against which to verify a greedy depth-two gap.

These properties make pi-Bench a strong proactivity and memory benchmark, but
not a direct test that non-myopic information gathering beats greedy
clarification.

### Registered future use

Keep pi-Bench as the first new source to try after a mechanically valid
non-myopic wrapper exists. The minimal acceptable route is:

1. use only the 30 dependency-final tasks;
2. freeze a two-action budget over prior-session/workspace inspection and one
   targeted clarification;
3. construct beliefs without exposing current hidden intents to the policy;
4. use prerequisite artifacts and official user responses as external
   observations;
5. define an endpoint from hidden-intent coverage plus available deterministic
   tool checks;
6. prove a strict greedy-versus-depth-two opportunity on a frozen development
   cohort before any OpenRouter call;
7. compare paired myopic, non-myopic, history-ablated, and matched-compute
   controls.

The later value-blind split and five-case mechanics audit found that the
dependency signal is real but not canonically observable. Final requirements
strongly echo predecessor hidden intents, while released predecessor
requests/titles/descriptions do not recover those inherited preferences. The
runner resets each session, `depends_on` is used for evaluation aggregation,
and no reference trajectories or memory snapshots are released.

**Updated decision:** close the exact target-blind pi-Bench wrapper. Revealing
predecessor hidden intents as memory would leak latent annotations; generating
our own prior memory would create a policy-dependent initial state. The
opportunity/development/holdout values remain sealed for a genuinely distinct
future end-to-end memory experiment.

## ClarifyBench

The later ClarifyBench source audit inspected release commit `a85d4f9` and all
604 bundled records. The release has 214 ambiguous, 242 explicit, and 148
infeasible examples. It contains realistic tool chains and 566 records with
fixed follow-up requests, but zero alternative latent worlds, zero
question-conditioned answer maps, and zero native tool-call turn annotations.

The harness enumerates every follow-up before interaction. Its main loop asks
the LLM user simulator for clarification responses but neither calls the
simulator's dynamic next-request method nor advances its turn state. The data
loader instead defaults every unannotated tool call to turn one. The repository
reports EVPI compatibility fields but does not include a SAGE/POMDP
implementation.

**Decision:** close the released ClarifyBench snapshot as a headline
non-myopic BED route. Converting it would require inventing alternative worlds
and answer transitions rather than evaluating a native released structure.
See `CLARIFYBENCH_SOURCE_AUDIT_RESULT.md` and audit SHA `02e4942a`.

## CRA-Bench

CRA-Bench has 250 underlying user/target worlds repeated across easy, medium,
and hard variants. Profiles and targets are constant within each triple. The
hard split has a two-turn patience budget, making its conversational
recommendation surface structurally attractive.

The current release is not executable as described. It contains task files,
hidden user profiles, visible recommender profiles, and 244 exact target
products, but no user-simulator code or prompt, retrieval runner, product
catalog, catalog reconstruction/filtering script, reference policy, or linked
paper. The exact products and metadata occur only in evaluation-only fields.

**Decision:** do not use evaluation targets as a closed policy support, because
that leaks the benchmark target pool. Close paid work until the official runner
and catalog are released; then audit the hard split for a target-blind
greedy-versus-depth-two gap before model calls. See the hash-pinned
`cra_bench_source_audit` artifact.

## DiscoverLLM

DiscoverLLM is the strongest released source of genuinely path-dependent
semantic state found in this audit. Its LLM-generated intent hierarchies have
median depth five in all three domains; initial states contain 22--27 hidden
nodes on average. Parent discovery controls which descendants can be updated
and articulated by the user simulator.

The public preference dataset nevertheless cannot test non-myopic BED directly.
The official synthesis launcher fixes `window_size` to zero, and all 2,642
recoverable committed transitions reproduce the released score as immediate
awareness gain minus the documented `[0,1]` token penalty. Per-domain
score/immediate-gain correlations are `.9974`, `.9985`, and `.9991`.

Flattening also removes every unchosen candidate's post-state and future
trajectory. Only the selected action's next state is recoverable from the next
turn. The release has no prior over mutually exclusive latent worlds, so it is
an intent-discovery process rather than a complete BED environment.

**Decision:** close direct replay of the released preference scores. Keep the
simulator as a future construction substrate only if a prospectively frozen
extension adds alternative latent worlds, positive-window rollouts, shared
myopic/lookahead actions, and an independent endpoint. See
`DISCOVERLLM_SOURCE_AUDIT_RESULT.md`.

## IG-clarifier

The official release trains a clarification policy with a single-turn
information-gain reward:

```text
avg_log P(G* | T(x,Q,A)) - avg_log P(G* | T(x))
```

Its required examples are user-provided `prompt`, `data_source`, and
`reward_model.ground_truth_clean` columns. The repository explicitly labels
itself a code-only release: task construction and evaluation are left to the
user, training and evaluation data are absent, the compatible `verl` checkout
is absent, and the strict user-simulator service clients are omitted. In
particular, it does not ship the tau-Bench environment, an executable
trajectory evaluator, alternative latent worlds, or a non-myopic policy
comparison.

**Decision:** close direct use. IG-clarifier is relevant evidence that semantic
clarification rewards can be trained, but applying it here would require us to
invent both the sequential BED task and its evaluation data.

## OPEN

OPEN uses an LLM to name and verbalize domain features, then performs greedy
one-step EIG selection with a particle-filtered Bradley--Terry model over a
fixed ten-feature space. The paper explicitly describes each interaction as an
ad-hoc greedy posterior update and lists fixed feature space as a limitation.
Its human-study endpoint is prediction accuracy on 15 pairwise news choices;
the paper says code and anonymized data will be released but provides no linked
release.

**Decision:** close as a non-myopic substrate. OPEN is a strong related-work
baseline for combining LLM semantics with classical BOED, but the released
formulation does not contain path-dependent support or delayed information
value.

## BIRD-Interact

Mini-Interact is a genuine semantic interaction benchmark: 300 SQLite tasks
contain 733 critical ambiguities, and its user simulator answers arbitrary
clarification questions from hidden ambiguity annotations and ground-truth SQL.
However, the public task file withholds all solution SQL and tests, has no
follow-up phase, and releases one interpretation per ambiguity term. Across
tasks there are 15 same-database repeated labels with different snippets, but
within a task zero terms have multiple released interpretations. There is no
prior over mutually exclusive SQL intents or world-conditioned response table.

**Decision:** close direct benchmark replay. Preserve a BIRD-derived
construction as a distinct future route: prospectively generate coherent
alternative SQL-intent worlds, freeze their prior, and use BIRD's simulator and
SQLite endpoint as semantic transition and validation machinery. This must be
reported as a new construction and pass a zero-cost manifest plus capped
mechanics gate before efficacy calls. See
`BIRD_INTERACT_SOURCE_AUDIT_RESULT.md`.

## Portfolio Decision

- **Headline:** continue to concentrate on an LLM-native result. The current
  strongest evidence remains tau-Knowledge V3.1: significant semantic ranking
  links and directional, underpowered endpoints.
- **CA-BED follow-up:** the queued open-ended/categorical Detective revisit is
  now closed at its zero-cost source gate. Each suspect has only one story
  written for the canonical role and the release provides no counterfactual
  story or response map under alternate murderers. Richer answer text would
  preserve the same simulator mismatch rather than define new worlds. The 24
  reserved cases remain unused.
- **Supporting result:** retain RockSample as exact-verification evidence only.
- **Next external source:** retry RegretBench when its official environment is
  released.
- **Future semantic-state substrate:** DiscoverLLM has the right path-dependent
  LLM state, but its released preference scores are exactly one-step and its
  counterfactual candidate states are absent. Any use must be a new,
  prospectively gated BED construction rather than a dataset replay.
- **Future semantic SQL substrate:** BIRD-Interact has natural clarification
  actions and executable SQLite endpoints, but no released alternative-world
  prior. A BIRD-derived LLM-generated intent construction is admissible only as
  a new, separately gated environment.
- **Future memory benchmark:** pi-Bench remains relevant, but its exact
  target-blind BED wrapper is closed for lack of a canonical prior observation.
- **Closed for now:** EComAgentBench's clarification-only graph, the released
  ClarifyBench scripted-world interface, CRA-Bench without its runner/catalog,
  the exact pi-Bench dependency wrapper, IG-clarifier's code-only single-turn
  interface, and OPEN's fixed-feature greedy BOED formulation.
- **Budget:** no paid calls and no OatML cluster work were used in this audit.
