# DiscoverPhysics Fixed-Initial Branch Replication

## Status

Frozen before any new branch response. This is a branch-support isolation,
not a rerun of the closed balanced full-tree replication.

## Fixed Question

The independently generated structured-V3 initial support selects D myopically
and produced a strong B-versus-D non-myopic gap. Its unconstrained branch
support failed the same-root fixed-B control. A later fresh balanced full tree
passed every branch breadth gate, but its newly generated initial support made B
myopic and therefore removed the non-myopic question before endpoint access.

This experiment holds the public structured-V3 initial support and its induced
branch partition fixed, then asks whether one fresh balanced set of
branch-conditioned LLM hypotheses adds endpoint value under modular inference.
It tests branch-generation robustness conditional on a fixed initial belief. It
does not test full-tree generation robustness.

## Frozen Source

Load only `initial_support` and `branches` from:

```text
results/nonmyopic/discoverphysics_dark_matter_structured_replication_v3/
discoverphysics-dark-matter-structured-replication-v3-20260728T063000Z/
MODEL_FROZEN.json
```

Required SHA-256:

```text
473cf5c883929a2cf8b6d862bebf69e1bb6b6401bea8c7b0edba955e847b7e1d
```

The source refresh supports are ignored. The initial support's frozen immediate
EIG ordering must still select D.

## Fresh Model Calls

Use `openai/gpt-5.4`, reasoning disabled, requested temperature `0`, strict JSON
Schema, and the already validated default OpenRouter route.

Make exactly eight scientific calls in one batch: one for each of two branches
under roots A, B, C, and D. There is no preflight and no newly generated initial
support.

Each response must:

- parse and compile without repair;
- choose a continuation different from its root;
- contain exactly two hypotheses in each of NE, NW, SW, and SE;
- use at least three geometry types; and
- differ from the fixed initial support.

Project each response's regional mass to the disclosed prior
NE/NW/SW/SE = `.4/.3/.2/.1`, preserving the LLM's relative within-region
weights. Do not add, delete, repair, or relocate a hypothesis.

## Fixed Policy

Condition exactly within the fixed initial and generated branch supports. Cut
component-level likelihood feedback and predict with:

```text
0.95 * fixed_initial_component_prediction
+ 0.05 * fresh_branch_component_prediction
```

No cap, grid, temperature, region-specific mixture, support edit, or endpoint
fit is allowed.

## Phase A Gates

All must pass before endpoint access:

- source file hash exactly matches the frozen hash;
- source immediate EIG selects D;
- exactly eight accepted responses;
- `http_attempts == requests + retry_count`;
- zero reasoning tokens and zero forced exits/finals;
- run cost at most `$0.25`;
- all eight responses satisfy parsing, compilation, breadth, and diversity;
- all eight supports differ from the fixed initial support;
- at least three roots have branch-distinct supports;
- B's two branches choose distinct continuations;
- modular trajectory risk selects B; and
- B reduces modular internal trajectory risk by at least `10%` versus D.

Failure closes this sample before the physical endpoint. There is no retry,
fallback, alternate initial support, or second branch sample.

## Untouched Endpoint

Only after writing and hashing model and policy files:

- map seeds `24740--24755`;
- noise seed `24756`;
- bootstrap seed `24757`;
- 384 maps, 96 per region;
- disclosed regional prior `.4/.3/.2/.1`;
- 8 root and 4 continuation samples per map;
- 10,000 region-stratified bootstrap samples; and
- modular non-myopic B, modular myopic D, modular random A, and same-root fixed
  B controls.

## Scientific Gates

All must pass:

- B reduces fresh-map MSE by at least `10%` versus D;
- paired interval lower bound for `MSE(D)-MSE(B)` is positive;
- B reduces MSE by at least `5%` versus random A;
- modular B reduces MSE by at least `1%` versus fixed B;
- paired interval lower bound for `MSE(fixed B)-MSE(B)` is positive; and
- routed retained-union nearest-support risk improves by at least `5%`.

Any failure is a replication null. Do not inspect alternate component masses,
regional weights, endpoint subsets, or another branch sample.

## Accounting

- Expected OpenRouter calls: `8`
- Projected cost: `$0.11`
- Hard run cap: `$0.25`
- Authenticated balance before freeze: approximately `$8.60`
- Reserve: none
- OpenRouter only; no OatML, Slurm, SSH, or cluster use
