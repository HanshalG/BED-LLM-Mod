# Debug-BED: condition on what the policy actually sees

## Decision

Do not relaunch SWE-smith V3 or purchase debugging calls. Its terminal privacy
failure did not measure planning opportunity, but fixing retrieval alone would
not establish a valid LLM-versus-oracle comparison. This is a protocol analysis,
not a new benchmark result or a claim that debugging cannot work.

The previously banked CI-Repair horizon null and R2E structural null remain
closed. ChemBench's failed proposal atlas also remains closed; its clean-response
semantic failures cannot be explained away by transport errors.

## Unresolved assumption in the old debugging construction

The old execution mechanics protocol places a uniform prior over subsets of
two to four patch hunks. Its future LLM interface would receive redacted code
and test context, but the protocol does not establish that this context is
identical across those worlds, or condition the reference prior on it.

Let W be the private defect world, C the initial public context, and H the paid
probe history. The relevant belief is p(W | C,H), not p(W | H). If the exact
visible program differs in each enumerated world, a reference observer with
the candidate-world dictionary can already identify W from C. An unconditional
status-matrix planning gap could then measure uncertainty the reference should
never have had. Conversely, identical redacted contexts may preserve genuine
uncertainty. Neither case has been measured on new tasks here.

This does not imply a real LLM can instantly understand code. It separates
epistemic uncertainty from resource-bounded program understanding. An LLM
advantage due to efficient understanding is a legitimate computational claim,
but cannot be described as a Bayes-optimal information gain from a context that
already identifies the finite reference world.

A second issue is population alignment: the mechanics prior is uniform over
subsets, whereas its selected released truth is the full subset. Prior-average
entropy and performance conditional on the full subset are distinct estimands.
Neither substitutes for a paired evaluation under the stated deployment law.

## Implemented boundary

`core/research_public_context.py` now conditions exact rational finite-world
weights on deterministic public-context identities before planning. It rejects
missing/extra worlds, negative or inexact weights, malformed identities, and
zero-probability observed contexts. It preserves nonuniform relative weights
and exposes when visible context leaves only one possible world.

The caller must bind every public field to the context identity and establish
that the policy really received those fields only. This helper is not a privacy
certificate, not a stochastic-context likelihood model, and is not connected
to a source runner. Its result never grants scientific or paid authorization.

27 focused tests across this helper and the existing headroom check pass in
0.44 seconds; scoped lint passes. These include explicit common-context,
partial-context and world-identifying-code fixtures. No source tasks, patches,
new outcomes, containers or model responses were opened.

## Concrete successor requirements

The defensible debugging candidate is shared public program/system context with
hidden runtime causes (configuration, state, or external component behavior),
not a hidden finite index that visible patched code immediately reveals.
Before selecting any cohort, its source contract must specify:

1. The joint distribution of hidden causes and public context, including initial
   logs and symptoms. Condition reference weights on all initial observations.
2. Fixed held-out behavioral predictions as the terminal scientific target.
   Fault-name entropy is only a diagnostic, not the primary score.
3. Executable probes whose outcomes depend on the actual hidden world. Every
   probe available to the LLM is available to controls at the same cost; no
   invented unlock rule or hidden reference library withheld from controls.
4. Executable LLM-generated causal explanations evaluated by numerical code,
   initially refreshed only after real observations and frozen within planning.
   Free-form model-predicted likelihoods do not become trusted evidence.
5. A full-budget reference after public-context conditioning, then ordinary
   receding h1/h2/h3 under identical total budgets. Two prospective 5% adjacent
   gains need at least 9.75% total headroom; headroom alone is not a pass.
6. Receding open-loop, productive compute-matched myopic, random and
   history-blind proposal controls, with common random numbers and sealed
   endpoints. A separately labelled thinking baseline does not replace them.

This is a specification for choosing a new source, not an assertion that an
existing benchmark satisfies it. No new source cohort or paid experiment is
authorized by this document. The next useful step is one concrete executable
source satisfying this contract, or an explicit rejection of this candidate;
another metadata-only cohort or revised SWE-smith salt would not resolve it.

## Accounting

Zero model calls and $0 new spend. Authenticated account observation remains
credits 245 / usage 220.376693994 / balance 24.623306006; London Sept8 ledger
spend is zero. No cluster use, runtime/cache modifications or automation change.
The research goal remains incomplete; there is no new positive LLM result.
