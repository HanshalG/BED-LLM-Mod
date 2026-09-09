# Herb backend: source-feasible, not runtime-qualified

## Pinned source

- Herb.jl:04de909d50077c6174bfe8940163d64c684b850f
  https://github.com/Herb-AI/Herb.jl/tree/04de909d50077c6174bfe8940163d64c684b850f
- HerbSearch.jl:c7abd55473593d43e8a2cbbcf754ba27bb15e397
  https://github.com/Herb-AI/HerbSearch.jl/tree/c7abd55473593d43e8a2cbbcf754ba27bb15e397

Read-only no-checkout clones are in /private/tmp/bed-herb-source-audit and
/private/tmp/bed-herbsearch-source-audit. No benchmark data, package scripts or
model calls executed. Julia is not currently on PATH. Root Project.toml declares
Julia1.10compatibility and HerbCore/Grammar/Constraints/Interpret/Search/
Specification1.x. A compatible resolved environment is not yet pinned or tested.

## Findings that matter for BED

The source exposes ProgramIterator, grammar/root-type/depth/size controls and
freeze_state. These support a bounded multi-candidate collection interface in
principle. The synth wrapper returns immediately at the first complete match;
it is inappropriate as a posterior-support builder without a separate collector.
Its stopping checks occur after evaluation, so hard runtime and exact work limits
also need an outer process guard, not merely max_time/max_enumerations keywords.

CostBasedBottomUpIterator assumes additive rule costs. It has an optional
program_to_outputs callback for observational-equivalence pruning, grouped by
return type. Importantly, the constructor defaults to nothing, which bypasses
the equivalence check; its prose documentation is less clear. Do not claim that
the default necessarily destroys uncertainty. Supplying demonstration-only
output signatures can suppress alternative intermediate programs from its bank.
An emitted root can still be yielded despite a failed bank insertion, so this is
not a claim that every equivalent root is automatically omitted. The underlying
integration risk remains: future distinguishing compositions may be unavailable.

Herb's documentation permits custom interpreters and a symbol table. This avoids
having to execute untrusted host Python, but does NOT prove that our Hodel
higher-order callable, row/grid/object and variable-size semantics have been
ported. The library is a candidate backend, not a drop-in full-language solution.
Its static additive-cost iterator is also not the unpublished Narcissus engine.

## Exact uncertainty regression

Two distinct programs, identity and horizontal-coordinate reversal, agree on
the public demonstration[[0,0]]. They disagree on query[[1,2]]. With equal
weights, normalized whole-grid Brier risk is.25before the query and0after it.
Merging them on the demonstration signature gives predicted risk0, but expected
loss.5under the original two-world distribution. This is an exact engineering
fixture, not an empirical depth gain, a new benchmark or a Herb runtime test.

The test exercises the existing condition_programs and mixture_scores functions.
They retain two distinct canonical programs with weights[.5,.5]. Two tests pass
in.10s; result is BED_SYNTHESIS_EQUIVALENCE_AUDIT_20260909.json. No task labels,
new candidate execution, or model calls were needed.

## Required integration contract

1. Consume the iterator with an explicit bounded loop, freeze each candidate,
   and collect multiple compatible programs. Never route through first-match
   synth. Count attempted candidates, valid candidates, failures and retained
   distinct programs separately. Do not confuse search depth with BED horizon.
2. Set program_to_outputs=nothing initially. Dedupe only canonical identical
   programs. Any future stronger equivalence must preserve relevant predictive
   distributions and their mass, not merely observed fit. Account for all public
   query/target inputs and future support changes before asserting equivalence.
3. Keep history-conditioned LLM search costs separate from the declared numerical
   prior/likelihood. Reusing an observed-data heuristic as though it were an
   independent prior risks double counting. Search truncation remains a source
   of approximate support, even with nonzero weights on every rule.
4. Bridge candidate ASTs to the existing isolated Hodel evaluator without reading
   source generators or hidden outcomes. Preserve resizing, objects, row
   intermediates, higher-order functions and DAG semantics. Do not fall back to
   only first-order same-size operations to make the adapter pass.
5. Reset iterator state per simulated branch; pass identical public histories,
   budgets and randomness to actual and simulated updates. A mutable shared
   grammar/bank must not leak information across branches. Bound CPU/memory/disk
   independently of model-call accounting.

## Next step and limits

The next executable step is a pinned isolated Herb runtime plus a small
cross-language candidate/interpreter test that includes callable intermediates,
resizing and the ambiguity regression. Do not install the whole benchmark suite
or launch another paid cohort merely because the API exists. If the full-language
bridge fails, bank the actual incompatibility before choosing another backend.
After runtime qualification, compare proposal-guided versus uniform search on
public-only engineering histories with equal work, then prospectively qualify
fresh predictive transfer and branch fidelity. None of these substitutes for the
final paired non-myopic depth result.

Source hashes:
- Herb Project.toml:668e39a0b432bc69613d94a98729ca96f897fea7385b1d00a4c58fab71be655c
- interpreter tutorial:88ed49dd3eb86fb25d8795d6eaa71634180cbc1261f2cbb3e24a5f6225711b28
- search_procedure.jl:7770ab19a61d6f11c5a0b44c8d71b95c78d23b4ee870df2e4a8c67304f83e9ac
- costbased_bus.jl:7aa40f06e8e1de48427b552879a1eabe5aadd130a520b3cae4bfa7401d89aebd
- program_iterator.jl:2bc4a2c12d6a137e3e1d722b50002b5ff2a2fb1941692ee8fcfcd0179b744edb

Authenticated OpenRouter state unchanged; no paid call authorized by this audit.
Both the preceding and current turns made progress. Full research goal remains
active and unachieved. No cluster or automation changes.
