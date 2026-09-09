# Semantic environment decision after the decomposition null

## New primary-source evidence

[SYNTRA](https://arxiv.org/html/2509.17393v1) studies program synthesis with visible
test inputs. It groups program output vectors, chooses queries greedily, and uses
an LLM to supply pseudo-labels that eliminate hypotheses. Its formal finite-class
setup assumes realizability. Its Playgol description uses five examples per task;
MBPP+ provides natural-language specifications and larger input sets. These are
relevant sources, not evidence for non-myopic Bayesian planning or reliable
open-world transition probabilities. Replacing a real observation with an LLM
pseudo-label would repeat our previous teacher-calibration problem.

[The original Playgol paper](https://www.ijcai.org/proceedings/2019/0841.pdf)
describes 94 real-world string-transformation tasks with ten examples each. That
is not the same shape as the SYNTRA release inspected here. Do not conflate versions
or assume that a paper's larger example bank is present in a derivative repository.

## Verified released-source contract

Pinned [SYNTRA repository](https://github.com/klee972/SYNTRA/tree/40d0bdfac9a1dade0afe49fa8308fadff78035c7),
commit 40d0bdfac9a1dade0afe49fa8308fadff78035c7. Recursive GitHub tree returned
truncated=false; no license-named file was present. Reuse/redistribution rights need
resolution before adopting its code or redistributing data. The paper's license
must not be assumed to license the repository.

The tested source audit downloaded pinned files, verified their hashes, and emitted
only aggregate schema/cardinality metadata. Full JSON payloads, including label
fields, were parsed in local memory. This is explicitly not a sealed-data loader;
no label values, program bodies or task text were shown to the model, no source code
was executed, and no task predictions were scored. Do not later describe this as
never having fetched the data. Only metadata and our audit implementation are banked.

| File | Rows | Released split | Important limitation |
|---|---:|---|---|
| playgol_v2.jsonl | 325 | 3 train + 2 test | 322 rows have five distinct inputs; three have four |
| mbpp_plus_51_cases.jsonl | 323 | One train + 50 test in list literals | Count only; distinctness and semantic ambiguity unverified |

Playgol has 325 unique names/IDs but one duplicated input panel. Names are not proof
of independent task semantics. There are zero rows with six distinct inputs. Exact
hashes and counts are in SYNTRA_SOURCE_CARDINALITY_AUDIT_20260909.json.

## Why this changes the next action

With one initial example and at least one disjoint endpoint, the five-input Playgol
release leaves at most three query candidates. With three equally costly queries,
the exact terminal posterior after querying them all is order-independent. Thus
it cannot provide a classical terminal-information advantage for ordinary h1/h2/h3
under that allocation. This is a necessary cardinality obstruction, not an empirical
oracle experiment. An imperfect path-dependent updater could still show order effects;
we must not mistake them for measured structural information headroom.

Do not manufacture extra labels, use the same row as query and scored endpoint,
quietly shorten the horizon, change costs to force a gap, or claim the original
ten-example source has already been verified. The derivative Playgol source is not
adopted for a three-depth experiment.

MBPP+ clears only the example-count obstacle. Full specifications may largely remove
semantic ambiguity; meaningful remaining uncertainty and non-myopic opportunity are
not established. Its released code must never be executed unsandboxed. It remains
a source candidate, not an authorized paid alternative.

The existing ZendoWorld audit already identified a label-contract mismatch, teacher
counterexamples and unresolved reuse conditions. BoxingGym's generative tasks are
relevant numerical yardsticks, but their mere availability does not establish a new
irreducible semantic role for the LLM. Do not reopen those banked routes by renaming them.

## Next dependency

Find and validate the original larger human string-transformation bank or another
licensed source with multiple query alternatives beyond budget and separate targets.
Inspect only a prospectively designated development slice for task semantics. Before
any proposer call, establish a complete executable labeling contract, strong symbolic
baseline, and a prospective numerical opportunity test. A context-rich environment
is a hypothesis about LLM suitability, not a result; retain paired myopic/width and
random controls plus fresh joint-prediction checks before depth evaluation.

No paid calls in this pass. Two audit tests pass in .09s; label-independent counts,
schema rejection and non-execution checks covered. Authenticated account remains
245/221.179955339/23.820044661, London conservative spend .76167686 and remaining
4.23832314 including the retained old .04 uncertainty. Previous turn was progress;
this turn adds primary-source and measured release evidence that rules out a tempting
but unsuitable three-depth setup. Full research goal remains active and incomplete.
