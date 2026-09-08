# INDUCTION canonical source contract: not qualified

## Decision

Complete the bounded source-only audit. The proposed reproducible non-tiny
generative source contract is NOT met by the inspected public artifact. Do not
build or launch its active BED adapter under the current authorization. This
is a source-contract decision, not a measured planning or LLM failure, and not
evidence that canonical INDUCTION concepts are simple.

No benchmark rows, reference formula rows, predictions, evaluation caches,
endpoint labels, or upstream executable code were loaded or executed. Reads
were restricted to pinned documentation/schema and Python source text/AST.
This is retrospective source inspection, not a prospective efficacy test.

## Verified Findings

Source commit: `c1f71f98623ab6e3513f820a8e52235d5b498694` in
[Concept Synth](https://github.com/SerafimBatzoglou/concept-synth/tree/c1f71f98623ab6e3513f820a8e52235d5b498694).

- The [provenance document](https://github.com/SerafimBatzoglou/concept-synth/blob/c1f71f98623ab6e3513f820a8e52235d5b498694/benchmarks/induction/docs/provenance.md)
  explicitly distinguishes the public generator from the private calibration,
  version-space diagnostics and downselection used for the canonical 775 rows.
  The previous static generator audit found ten fixed templates. Neither fact
  establishes a reproducible rich generative prior for the proposed experiment.
- The [schema](https://github.com/SerafimBatzoglou/concept-synth/blob/c1f71f98623ab6e3513f820a8e52235d5b498694/benchmarks/induction/schemas/benchmark_record.schema.json)
  requires `problemDescription.hiddenTarget.formula` and world target extensions.
  Passing a raw record, or merely removing model responses, would leak answers.
  Schema validation alone is not a policy privacy boundary.
- Train/test world identifiers exist in FullObs/EC. These are within-task fields,
  not evidence of an independent concept-family development/confirmation split.
  The inspected contract supplies no such split guarantee.
- `evaluate_fullobs` calls `_train_worlds`; absent train identifiers it uses all
  worlds. It checks exact extensions there. Its `correct` flag must not be used
  as a held-out predictive endpoint without a separate, explicit evaluator.
- The finite-model evaluator supports unary/binary predicates, equality,
  Boolean operators and quantifiers. Formula evaluation is recursive over finite
  domains; source inspection does not establish a practical depth-three planning
  runtime. The main prediction evaluator requires exactly free variable `x`.
- The parser also supports `implies` and `iff`; parser expressivity is broader
  than a restricted prompt language. A future adapter must freeze its allowed
  language explicitly rather than assume the parser enforces the prompt.
- The inspected FullObs/model evaluation path has no explicit operation/time
  cap. EC exposes a default 120000 ms completion timeout. This is not a safe
  universal budget for evaluating arbitrary generated formulas in a planner.
- Public world construction ignores malformed binary pairs and out-of-domain
  facts in some paths. Any future adapter needs strict validation rather than
  silently relying on these permissive conversions.

## What This Does And Does Not Rule Out

An independently frozen empirical task distribution over canonical formulas
could be scientifically legitimate without reproducing their generation. It
would be a DIFFERENT contract: a finite released suite, not fresh samples from
the original calibrated generator. That alternative has not been tested or
authorized here. It would need leakage-safe task-level splitting, a declared
formula prior, label-independent query/target worlds, and classical synthesis
controls with the same public information. Excluding those controls or hiding a
small truth library from them cannot establish LLM necessity.

Membership queries on new finite structures are a plausible new active
extension, not a released sequential BED protocol. Their distribution, noise,
budget and terminal loss must be specified before observing planning gains.
Source validity alone would not imply either meaningful non-myopic opportunity
or depth monotonicity. Keep the previous scene, ACRE and Number Game nulls closed.

Do not replace the missing source contract with another hand-picked small
grammar or select released tasks based on depth gains. Do not resume SciLaws
micro-optimizations merely because this alternative did not qualify. The next
architecture decision should explicitly compare the finite-suite contract
against returning to source-grounded model discovery; no new solver or paid
experiment is justified by this audit alone.

## Reproducibility

SHA256 of the exact pinned files read in this pass:

| Path | SHA256 |
| --- | --- |
| benchmarks/induction/docs/provenance.md | 53fb78b801a49d9bf8bb089672ba3cb717bf48aa1261417709d08eb0b6b966dd |
| benchmarks/induction/docs/release_contract.md | b6660fbb276f5f17de84d43ef446447de445d563b95e1587f64749bb70bd7891 |
| benchmarks/induction/docs/dataset_schema.md | 4b509c017f856ae37f28aabc069ef2739d0ae4cca8a4beeb5a4a9ca7f4731593 |
| benchmarks/induction/schemas/benchmark_record.schema.json | 01f494336fafc48efd79d1be1cfd3d40dd7481d80ef6b0a986bd854cb22ef1a6 |
| src/concept_synth/induction/evaluator.py | 9c68d961dc36d6ce3abaf1e3b1c90ebbc779bfc890e9332faf4d283907fe4bec |
| src/concept_synth/fol/model.py | c7d408c95442835c790509ea54b9de20835e6eb6fe717485c769b8b4d9497912 |
| src/concept_synth/sexpr_parser.py | 0424b34e9af36cc613d2c3818b8e6615e59b22bb791a624bdbe14212e90516d6 |

Authenticated credits/usage/balance: 245/220.376693994/24.623306006.
London Sept8 ledger validated: account-wide spend0, remaining5. No model calls,
cluster actions or scientific runner launches. Documentation-only change;
no production behavior changed. Automation stays paused; overall goal unfinished.
