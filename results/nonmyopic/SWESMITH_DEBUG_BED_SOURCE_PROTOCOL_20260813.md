# SWE-smith Debug-BED source protocol

Date frozen: 2026-08-13

Status: **metadata-only source admission; zero model calls**

## Scientific target

Use DebugGym over SWE-smith composed software defects as an LLM-native
sequential BED environment. The latent state is a semantic set of possible bug
locations and mechanisms in a real repository. Actions are executable
diagnostic experiments: focused tests, breakpoints, stepping, and explicit
variable inspection. Observations should regenerate and reweight the LLM's bug
hypotheses and determine which experiment is useful next. The external endpoint
is a submitted patch scored by the released fail-to-pass and pass-to-pass tests.

The LLM is necessary here because neither the repository-scale hypothesis
space nor semantic observation likelihoods are enumerated by the environment.
However, composed defects do not by themselves prove a horizon advantage. This
source audit can authorize only a separately frozen execution mechanics gate.

## Immutable sources

### DebugGym

- repository: `https://github.com/microsoft/debug-gym`
- commit: `cc3fe3ef4ce08919e522eb00ea1bea5689f3b53e`
- tree: `54b04c0313b66d72c7f285dc255d2693164a7193`
- license: MIT
- SWE-smith environment SHA-256:
  `a3fe5cb78df49df5742611bf33e6587ff943dc64cd6dcc90caa6a256bf92e3e2`
- repository environment SHA-256:
  `705faa2e83dac1336c7a6f62aab839a3b633a77a722b4f09c50ee583c2740967`
- PDB tool SHA-256:
  `ffd1453415ee0588bf72c62c3adb8ea238f9d2c004352aa884d8b947e79bc60b`
- eval tool SHA-256:
  `6012e0ec5fc5108924dc373fd5e5908b2fedf6c4e9fb6dae98bf26faaae7f2ff`
- view tool SHA-256:
  `f2948627292b1030f457ef64da437da01a8d23909ec426d0ed1832c6b3843ee1`
- edit tool SHA-256:
  `5362752ad45671a2933037641a944a441099e531337ee89298e13e000e130434`
- submit tool SHA-256:
  `a424d295b764269d7ff0a382b432664917b561c58c57c211bfde26efde1d9c50`
- official split-config SHA-256:
  `b5f8e0e12f96adb46f79769a29c603c30dc92331dcdca9cb5c9d6feffb287b14`

### SWE-smith data

- repository: `https://huggingface.co/datasets/SWE-bench/SWE-smith`
- revision: `699b53400d3855206a0fbf3ff4beaf1a52f4f232`
- license: MIT
- immutable population: eleven Parquet shards

The audit binds every shard's path, byte size, and LFS SHA-256 object ID in its
implementation. Moving revisions or partial shard sets are forbidden.

## Metadata-only gates

The source passes only if every gate holds:

1. DebugGym commit, tree, and all bound code hashes match exactly.
2. The Hugging Face revision and all eleven shard path/size/LFS bindings match.
3. The Parquet population has one exact schema and at least 40,000 rows.
4. `instance_id` is nonempty and unique; `image_name` is nonempty.
5. At least 2,000 tasks have `combine_file` or `combine_module` in their exact
   instance ID, with both composition types represented by at least 500 tasks.
6. Every composed task has at least one fail-to-pass and one pass-to-pass test
   identifier, measured only by list length; names remain sealed.
7. The released DebugGym code loads the exact data revision, applies the bug
   patch to a fresh repository image, supports persistent PDB experiments,
   executes official tests, and scores fail-to-pass only when all pass-to-pass
   tests pass.
8. The released environment removes the upstream remote, so policy execution
   cannot retrieve later source history.
9. The official DebugGym `train-789` and `test-125` instance-ID lists are
   present, unique, disjoint, and subsets of the immutable population after
   exclusions. They remain development and confirmation respectively.
10. From composed tasks outside those official lists and exclusions, a
    deterministic hash ordering supports 8 mechanics and 64 opportunity tasks.
    All populations are complete and disjoint.

Public artifacts serialize only aggregate counts, exact schema names, mutation
type counts, repository/image counts, and hashes of complete ordered ID lists.
They serialize no individual instance ID, problem statement, patch, test name,
test output, repository source, gold fix, or endpoint.

## Frozen split

Composed IDs outside official development/confirmation and exclusions are
ordered by `SHA256("swesmith-debug-bed-v1|" + instance_id)`:

| Split | Count | Access after source audit |
|---|---:|---|
| mechanics | 8 | only after passing source audit and pushed boundary |
| opportunity | 64 | sealed until mechanics protocol |
| development | official `train-789` | sealed |
| confirmation | official `test-125` | sealed |
| reserve | remaining eligible composed tasks | sealed |

## Required execution mechanics successor

A source pass authorizes only a separately frozen eight-task zero-model-call
mechanics gate. Before any LLM call it must establish:

- Linux container and native PDB completed-handshake replay in two fresh arms;
- identical initial eval, breakpoint, stepping, and variable-query outputs
  after removal of explicitly frozen runtime-only fields;
- test names and gold patch remain inaccessible to policy prompts;
- at least three valid diagnostic experiments before editing;
- an adaptive dependency where one experiment's output changes the valid or
  useful next experiment;
- a finite source-derived bug-hypothesis bank used only for opportunity
  verification, never exposed to the policy;
- an exact depth-two diagnostic root differing from a compute-matched receding
  myopic root with positive first-link utility on at least five of eight tasks;
- nonsaturated patch endpoints and paired common-random-number execution;
- dynamic-support, fixed-support, compute-matched myopic, and random controls.

If mechanics passes, a separate serving protocol must gate strict semantic bug
hypotheses, answer-obedient likelihoods, support regeneration, and cost before
opening opportunity or development. Any failure closes this exact construction.
There is no task replacement, subset repair, threshold relaxation, or endpoint-
informed prompt repair.

## Accounting

- OpenRouter calls: `0`
- OpenRouter cost: `$0`
- OATML cluster use: none
