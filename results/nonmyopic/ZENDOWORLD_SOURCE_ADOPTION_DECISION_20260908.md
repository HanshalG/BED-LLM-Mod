# Executable semantic induction: source adoption decision

Status: candidate retained, stock wrapper not adopted. No efficacy result, source
opportunity pass, paid permission, or reopening of a closed endpoint.

## Why change the direction

The Newton diagnostic suggests many stock known-law classification cells are cheap
to distinguish. More fixed small-bank depth studies do not address the missing LLM
contribution. Executable semantic rule induction is a better candidate for testing
proposal quality and future hypothesis discovery, but symbolic search must be a
real competitor rather than an intentionally weak placeholder.

Relevant primary sources:

- [Doing Experiments and Revising Rules (NeurIPS 2024)](https://arxiv.org/abs/2402.06025)
  already combines language proposals, belief revision and experiment selection.
  Merely adding an LLM proposer is not our novelty; anticipated discovery and
  controlled physical-query lookahead remain the research question.
- [ZendoWorld](https://arxiv.org/html/2607.08233v1) provides active visual concept
  induction and a uniform-PCFG symbolic baseline. Its typed scene rules make
  executable proposals plausible. A structured-text-only study is a distinct
  isolation variant, not a reproduction of its visual benchmark.
- [SciLaws-Bench](https://arxiv.org/html/2609.01552v1) distinguishes fixed real-data
  recovery from active queries in parallel worlds, with empirical residual-based
  noise. It is a possible later transfer target, not a verified Gaussian BED
  adapter or currently source-gated experiment.

## Pinned source and concrete check

Inspected [ZendoWorld source](https://github.com/ml-research/ZendoWorld) commit
`c18e24eae479a91f45fa0615d809f022c15281f1` in a no-checkout clone. No image corpus,
weights, game imports, model calls, Blender or Prolog execution were requested.
Selected text blobs were read through Git. The tracked tree has no filename
containing license/copying/notice; this is not a legal determination. Resolve reuse
terms before redistributing upstream implementation or data.

`ZENDOWORLD_SOURCE_CONTRACT_20260908.json` binds six inspected source hashes.
The new isolated reproducer executes only constructor and membership-label methods,
with a constant-True program and no-op program normalization. For a fixture whose
stored label is False, the constructor prints a mismatch warning but retains False;
the membership query on the same scene returns True. This demonstrates a wrapper
contract, NOT a measured rate of corruption in released data. Three focused tests
pass, including extraction isolation and rejection of a changed contract.

Further source inspection, not an executed end-to-end test:

- `zendo/states.py` calls the executable membership oracle for proposed scenes,
  but wrong guesses can also yield teacher counterexamples observed by players.
  These are distinct information channels. A BED comparison cannot silently add
  teacher-selected examples to a fixed membership-query budget.
- `initial_examples` selects positive/negative examples; `test_scenes` similarly
  balances classes. Do not inherit these as truth-independent random targets.
- `check_guess` can use model conversion/judging and semantic special cases;
  source correctness is not a held-out predictive metric.
- `DSL/vlp_dsl_symbolic.py` contains model-based parsing. Its name is not evidence
  of zero-call numerical likelihoods. The tensor DSL also needs strict type/error
  checks rather than accepting silent exception-to-False/zero fallbacks.

## Concrete next implementation boundary

Build an independently specified membership-only structured-scene prototype before
any model request. Do not port the game master or claim visual-benchmark results.

1. Public bounded scene schema and typed executable rule grammar. Canonical object
   order and valid relation constraints; malformed rules/scenes fail explicitly.
   One deterministic evaluator supplies every training and test label. No natural
   language judge, guessed-rule equivalence reward, or counterexample service.
2. Explicit public prior and complete prospective task-generation strata. Avoid
   outcome-screening for attractive horizons. Keep evaluation scenes fixed across
   policies and independent of the hidden rule. State whether initial examples
   are random or teacher-selected and include that selection in the model.
3. Test evaluator identities, permutation invariance, invalid-input rejection and
   exact tiny-space Bayesian updates against an independent enumerator. Then freeze
   a tractable source-only opportunity panel, including genuine receding h1/h2/h3,
   receding open-loop, random and compute-matched myopic controls at equal query
   budgets. Exact small-space success is mechanics, not evidence of LLM necessity.
4. Only with useful opportunity, freeze a larger held-out executable-proposal gate:
   history-aware, history-blind, explicitly shuffled-feedback and actual PCFG/search
   proposals; common numerical updates; sealed predictive targets; full rejection
   accounting. A proposal gate must show useful semantic coverage and calibration,
   not merely JSON validity or consistency on observed examples.
5. Only then compare real-history refresh with anticipated proposal transitions.
   Test transition fidelity on held-out histories before claiming lookahead over
   future discovery. Fixed-support planning remains an intermediate control.

This is an adoption specification, not yet a frozen experimental protocol: scene
grammar, priors, coverage, compute budgets and thresholds must be fixed and tested
before their corresponding responses. No current result authorizes paid calls.

## Project disposition

No change to earlier nulls or thresholds. The overall plan is unfinished; we still
lack a fresh calibrated LLM proposal advantage followed by paired non-myopic policy
efficacy and confirmation. Automation stays paused. No cluster use. This turn made
zero model calls and incurred no inference cost; no fresh account balance claimed.
