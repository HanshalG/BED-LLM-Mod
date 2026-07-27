# Pi-Bench Dynamic-Support First-Link Preregistration

Date: 2026-07-28

**Status: frozen before opening any development task value or making any model
request.**

## Claim Under Test

The experiment tests one narrow causal link:

> A first clarification question selected for its expected two-turn value can induce
> a better LLM-regenerated belief support after the user's reply, and that improved
> support can produce a better second clarification question than selecting the first
> question by one-step expected resolution alone.

This is not a full Pi-Bench agent score and does not claim that planning improves tool
execution. It isolates the part of sequential BED that must be LLM-native: generating
latent semantic requirement sets, mapping free-form questions to those requirements,
and regenerating the support from a path-dependent dialogue history.

## Frozen Source And Cohorts

- source: `https://github.com/Simplified-Reasoning/Pi-Bench`;
- commit: `383910b1698758a198b86037c63a111c8edc32ad`;
- source manifest:
  `results/nonmyopic/pi_bench_release_source_manifest.json`;
- manifest SHA256:
  `ccdf9211016d6c77eefc6cb3aae4e0324640c252b9d3aa17551ad158fc61594e`.

The chronological split in that manifest is immutable:

- mechanics: sessions 1-4 per persona, 20 tasks;
- development: sessions 5-10 per persona, 30 tasks;
- confirmation: sessions 11-16 per persona, 30 tasks;
- retained: sessions 17-20 per persona, 20 tasks.

The serving smoke uses the first mechanics task from each persona:
`Financier_task_001`, `law_trainee_task_001`, `marketer_task_001`,
`pharmacist_task_001`, and `researcher_task_001`. Their hidden-intent counts were
already opened during the source audit: 6, 2, 3, 7, and 3.

After this document is committed, development task values may be opened only to:

1. load the official visible context and private simulator state;
2. count initially `not_provided` hidden intents;
3. execute the frozen policies and endpoint.

A development task is eligible exactly when it has at least three initially
`not_provided` hidden intents. Every eligible development task is included. Fewer than
twenty eligible development tasks fails the opportunity gate. Confirmation and
retained values remain sealed until explicitly authorized below.

## Information Boundary

The policy receives:

- the current task's `intent.initial_input`;
- the released persona role, preferences, and long-term goals;
- the two-turn dialogue history generated within its own trajectory.

The policy never receives task title, description, objectives, checklist, hidden
intent contents or statuses, dependency labels, files, tool traces, or evaluator
outputs. Policy request construction accepts a public-task object that cannot contain
private fields. Before sending, a leakage audit rejects any policy payload containing
a private hidden-intent string that is not already present in the visible initial
input or persona context.

Private hidden intents are used only after a policy has selected a question:

- by the pinned Pi-Bench satisfaction and targeted-followup semantics to advance that
  trajectory;
- by a post-selection support-recall evaluator;
- to calculate covered intent indexes and normalized coverage.

Raw prompts, responses, task values, and hidden-intent text stay in an untracked
private directory. Public artifacts contain task IDs, hashes, candidate IDs, selected
IDs, integer intent indexes, scores, aggregate metrics, usage, and gates.

## Models And Reasoning

The BED generator, semantic mapper, user simulator judge, and post-selection evaluator
use `openai/gpt-5.4` through OpenRouter at temperature zero with reasoning disabled.
A provider seed is requested where supported.

The naive baseline uses the same model with medium reasoning enabled and asks one
clarification question directly from visible history. Reasoning is not used by any BED
policy. Transport retries are allowed and counted separately; malformed semantic
outputs fail closed and are not silently replaced.

The authenticated OpenRouter balance immediately before freezing this protocol is
`$28.332494594`. The full remaining balance is authorized by expected scientific
value. There is no fixed reserve and no artificial per-stage cost throttle. Every
accepted request must still report usage and cost to the project ledger.

## Shared Initial Belief And Actions

For each task, one target-blind model request generates:

- `W=8` equally weighted latent worlds;
- each world is a coherent set of 3-7 plausible hidden requirements;
- `Q=6` distinct candidate clarification questions.

Worlds and requirements are normalized and deduplicated. Questions must be specific
enough to resolve a recognizable requirement or coherent group of requirements.
Generic prompts such as "anything else?", requests to list every requirement, and
omnibus checklist questions are invalid. No true hidden intent is inserted into or
used to repair the support.

A separate target-blind semantic-map request applies the released targeted-followup
criterion to every question/world/requirement tuple. If a question matches one or more
requirements in a world, the simulated reply is those requirement texts in world
order. If it matches none, the reply is the world's first unresolved requirement,
matching the released simulator's first-unmet fallback.

Interface v6 encodes each local semantic match as a fixed seven-bit string for
requirements `R0` through `R6`. Bits beyond a pair's actual support are padding and
cannot denote a requirement. They are deterministically masked to zero and every pair
requiring that normalization is counted. Active-support bits, pair counts, and
bitstring syntax are never repaired. A padding-normalization rate above `0.10` fails
the serving/integrity gate as evidence that packed semantic mapping is unreliable.

For a world with `m` requirements, immediate utility is the number of newly resolved
requirements divided by `m`. Initial worlds are uniform.

## Depth-Two Rollout

Four of the eight initial worlds are selected without replacement from a deterministic
SHA256-derived seed. The same four world indexes are used for every root question.
For each of the resulting `6 x 4 = 24` branches:

1. append the root question and its simulated reply to visible history;
2. if the root resolved every requirement in that sampled world, assign terminal
   utility one and make no unnecessary regeneration request for that branch;
3. otherwise discard the old support and regenerate `W2=4` possible remaining-requirement
   worlds from that history;
4. generate `Q2=4` follow-up questions from the refreshed support;
5. semantically map every follow-up question to the unresolved requirements of the
   branch's original sampled world;
6. select the follow-up with maximum expected immediate resolution under the refreshed
   support;
7. score the selected follow-up against the original sampled world.

Step 6 is crucial: refreshed support that forgets the sampled world's remaining truth
receives no self-consistency credit. Terminal utility is the fraction of unique
requirements in the original sampled world resolved after the two questions. The
depth-two root score is mean terminal utility across the four common rollout worlds.

Up to 24 incomplete-branch refreshes may be packed into one physical request, but
every branch is parsed and scored separately. Packing cannot mix branch histories or
share generated support between branches. Skipped terminal-complete branches remain
in the root's four-world mean with value one.

## Policies

The paired policies are:

- `myopic`: root with maximum expected immediate utility over all eight initial
  worlds;
- `depth2`: root with maximum expected two-turn terminal utility over the four common
  rollout worlds;
- `random`: root sampled uniformly from the same six-question bank using frozen seed
  `24422`;
- `naive_thinking`: a medium-reasoning direct clarification question generated from
  visible history without the BED support.

Ties use the lowest canonical candidate ID. Myopic and depth2 consume the exact same
generated support, candidates, maps, branch refreshes, and tokens; myopic ignores the
terminal score when selecting. This is the matched-compute width control.

After the realized first reply, each BED trajectory independently regenerates eight
worlds and six follow-up candidates from its actual visible history, maps them on that
refreshed support, and asks the maximum-immediate-utility second question. The naive
baseline makes a second medium-reasoning direct choice. Random uses the same
myopic continuation rule after its random root so that it isolates first-question
quality.

Each policy starts from a fresh copy of the same official task state. User-simulator
model seed, policy presentation order, and random root are paired by task.

## Endpoints

Primary endpoint:

- official covered-intent fraction after two questions, counting the union of
  `provided` and `inferred` private indexes.

Primary first-link mechanism endpoint:

- semantic recall of the actual still-unresolved hidden intents in the refreshed
  support immediately after the realized first reply.

Secondary diagnostics:

- official covered fraction after question one;
- incremental coverage from question two;
- initial support recall of the actual hidden intents;
- root-choice disagreement;
- predicted depth-two advantage;
- realized coverage advantage;
- Spearman association between predicted and realized advantages;
- question uniqueness and invalid-question rate;
- support fingerprint changes across different first-question branches;
- parse failures, semantic retries, transport retries, reasoning tokens, tokens, cost,
  and wall time.

All comparisons are paired by task. Report per-task values, means, standard deviations,
paired bootstrap 90% and 95% intervals, and an exact paired sign/permutation test where
applicable. Ties remain ties.

## Gates

### Serving smoke

All five frozen mechanics tasks must complete. The gate requires:

- no private-label leakage into policy requests;
- exact requested world/question/branch counts;
- zero BED reasoning tokens and positive naive reasoning tokens;
- zero malformed accepted outputs;
- semantic-map padding normalization on at most 10% of mapped pairs;
- at least four unique valid root questions per task;
- nonidentical refreshed-support fingerprints for at least two histories on at least
  four tasks;
- myopic and depth2 select different roots on at least one task;
- finite endpoint and usage records for every policy/task.

Only implementation defects may be repaired after inspecting serving outputs. Any
prompt or scoring change creates a new version and requires another mechanics smoke.

### Development

Development runs once on every eligible frozen task. It authorizes confirmation only
when all integrity gates pass and:

- myopic and depth2 roots differ on at least 20% of eligible tasks;
- among changed-root tasks, depth2 minus myopic refreshed-truth-recall is at least
  `+0.05` on average;
- overall depth2 minus myopic two-turn covered fraction is at least `+0.03`;
- depth2 coverage wins exceed losses by at least two tasks;
- at least half of the observed coverage advantage is mediated by a positive
  second-question increment rather than first-question coverage alone.

No criterion may be changed after development values are inspected. A failed
development gate closes this Pi-Bench variant; it does not authorize prompt tuning on
confirmation.

### Confirmation

If development passes, run the unchanged protocol once on every eligible confirmation
task. The headline is positive only if:

- mean depth2-minus-myopic two-turn coverage is positive;
- the one-sided exact paired permutation test is `p <= 0.05`;
- mean refreshed-truth-recall difference is positive;
- wins exceed losses;
- depth2 is not below random on mean two-turn coverage.

Naive thinking is reported as a separate baseline, not a gate on the causal
depth2-versus-myopic claim. Retained tasks remain untouched unless a paper revision
explicitly preregisters their use.

## Stop Rules

Stop immediately on hidden-label leakage, source hash mismatch, unpaired simulator
state, malformed public/private separation, or evidence that the action space reduces
to a released finite intent menu. Stop after a failed serving or development gate.
Do not spend remaining balance merely to complete a failed grid.
