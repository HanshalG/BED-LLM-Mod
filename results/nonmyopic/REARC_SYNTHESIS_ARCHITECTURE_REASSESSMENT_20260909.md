# Reassessment: LLM guidance, verified search, then experimental design

## New project evidence

The eight-task feedback study remains closed at1/8initial coverage. A static audit
of all16saved responses finds reference/identifier failures in7batches;3batches
contain nested function expressions where the graph protocol required a named
argument. Other failures include forward references, repeated IDs, unsupported
NEG_THREE and an absent output step. No graph was repaired, executed or rescored
by this audit; target labels remain unopened. Three focused tests pass.

The custom graph serialization contributes avoidable difficulty. Nevertheless,
23initial candidate programs already returned valid but wrong grids. Therefore
changing serialization alone cannot be presumed to solve semantic induction.

## Literature that changes the next action

[HySynth, NeurIPS2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/1c9c85bae6161d52182d0fe2f3640512-Abstract-Conference.html)
uses LLM proposals to estimate a task-specific grammar distribution that guides
symbolic synthesis. Its central distinction is useful here: proposals can inform
search without being correct executable answers themselves.

[Narcissus, Aug26 2026 preprint](https://arxiv.org/abs/2608.25657)
retains proposal-tree context and recurring fragments, replaces invalid subterms
with typed holes, and keeps unmentioned grammar rules reachable. Search makes no
additional LLM calls. Its reported ARC improvement is a synthesis result, not
sequential BED. This is a recent preprint, not our replication. Its paper-linked
repository https://github.com/Herb-AI/Narcissus returned404 on this inspection;
do not claim its implementation has been verified. A positive rule floor also
does not prove finite-budget dominance or monotonic held-out risk.

[ConceptSearch](https://arxiv.org/html/2412.07322v2) uses iterative program search
with richer guidance than pixel mismatch; its prompts include programs and their
outputs, and its initialization uses training-task solvers. Our two-call repair
study is not a reproduction. Do not import target-task solver examples into a
new study. Its larger search effort must be accounted for, not compared against
our tiny call budget as though everything else were equal.

These sources support a different allocation of work, not a promised positive
result: LLMs suggest structure; a grammar-aware search engine constructs and
checks candidates. Avoid another unmotivated increase in reasoning or repair calls.

## Source feasibility audit

HySynth source is available at https://github.com/shraddhabarke/hysynth,
commit f839739d435880fcd4ee4bdf134cee3e7fd2bd9d, inspected in a no-checkout clone
at /private/tmp/bed-hysynth-source-audit. No dataset or model generations executed.
The ARC implementation uses an object-filter/transform language, not our full
Hodel grid DSL. It loads training files and cached generations by task ID; those
entrypoints cannot be called inside a sealed branch updater. Costs are computed
from grammar probabilities, with Laplace smoothing visible in the inspected code.
Do not indiscriminately repeat the claim that this version discards every unseen
rule. It needs an explicit public-history adapter and expressivity check, not
an import-and-run switch. No top-level license file appeared in the root tree;
resolve reuse terms before vendoring. Herb.jl's public library is available, but
its generic availability is not availability of the Narcissus implementation.

Inspected file SHA256:
- README.md: d29b27328696f6f756994db646a7571de052b903407e32ed20871e623b72281c
- compute_pcfg.py: 3de9bf8ec812a60a89ba3a00fe7699967472aa36f45f6f7b61fc4a5043e4570a
- run_synthesize.py: d8c1df98ae514f42506089709a776fa13bfe673983a2cccaf4f8d3a7bd3bb039
- dsl/v0_3/dsl.lark: 35d26f9b9492ab67dc980ce1c4e9f51cbe8c4537b50ed0ae413a10ff7b3f24ba

## BED-specific architecture requirements

Our proposed extension, not a claim from those papers:

1. Make the LLM output guidance or partial programs, not authoritative posterior
   probabilities. A fixed numerical update checks complete candidates against
   the observed history. Search guidance must not be silently treated as a
   data-independent Bayesian prior and multiplied by the same history again.
2. Keep a nonzero exploration path outside proposal fragments. Measure both
   retained predictive coverage and the budget lost to misleading guidance.
   Grammar reachability is not coverage within the actual compute cap.
3. Generate multiple history-consistent hypotheses, not the first fitting program.
   First-solution stopping may be adequate for synthesis but leaves no meaningful
   uncertainty over future query answers. Preserve failures and finite-support
   limitations instead of manufacturing confidence.
4. Treat proposal-guided search as the bounded belief-update operation. Simulated
   and actual steps use the identical public-history interface, cost limits and
   randomness treatment. A sampled world is persistent and separate from the
   candidate belief; it is never injected into that belief.
5. First test guided vs unguided search and answer-aware vs equal-call blind
   guidance on predictive transfer. Then test whether simulated updates rank
   queries correctly against independent actual updates. Only then test h1/h2/h3
   with the same fixed-target proper risk and equal-compute myopic/width controls.

The hoped-for non-myopic mechanism is a query whose answer makes subsequent
search more productive, improving the next belief and next experiment. If gains
are explained entirely by extra search, or one-step acquisition saturates, there
is no headline depth result. Predicted conservative policy improvement cannot
guarantee empirical monotonicity under model misspecification.

## Next executable gate

Inspect an existing grammar-aware search backend for full-grid, variable-shape,
higher-order support and safe public-history evaluation. Do not start by writing
another complete search engine or stripping the environment to pass a weak
baseline. Before new paid calls, use exact synthetic grammar checks and an
explicitly retrospective public-only guidance-vs-uniform diagnostic; neither may
open the closed target outcomes or rescue their gate. If a backend lacks the
required expressivity, bank that fact and choose a different backend rather than
claiming success on a narrower substitute. A fresh paid interface/cohort needs
its own frozen protocol after this integration gate.

No additional OpenRouter spend; London-day remaining4.0275717 at the authenticated
read. Previous and current turns are progress; the full goal remains unachieved.
