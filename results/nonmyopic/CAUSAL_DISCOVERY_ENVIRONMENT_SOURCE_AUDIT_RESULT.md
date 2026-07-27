# Causal-Discovery Environment Source Audit

## Decision

No audited release improves on the current LongVid/tau-Knowledge route for an
irreducibly LLM-native non-myopic BED result. No paid experiment is
authorized from this audit.

## CausaLab

The May 2026 paper describes the most attractive task:

- a new hidden SCM is sampled each episode;
- the agent sees prior records and chooses budgeted interventions;
- a DSL records its graph, equation, coefficient, and next-experiment
  hypotheses; and
- the endpoint tests transfer to a held-out reactor crystal.

However, the paper's linked repository,
`https://github.com/DylanZSZ/CausaLab`, currently returns `404`. The arXiv
source contains prompts and prose but not the SCM generator, episode seeds,
intervention implementation, or exact replay artifacts. Reconstructing those
details would invent a new benchmark rather than evaluate the release.

Even if the source appeared, the published 3--7-node mostly linear SCM
families admit classical causal-discovery baselines. CausaLab is a strong
interactive-science benchmark, but not yet evidence that an LLM must own the
hypothesis support.

Paper: `https://arxiv.org/abs/2605.26029`.

## ActiveACRE

The official `acre` branch is pinned at
`1a85018829f780d50a520cae74fb6940c4a65c04`.

Its generated game does not use the semantic ACRE rules as hidden mechanisms.
Each episode samples eight distinct objects and independently marks each
object active/inactive, rejecting only the all-inactive assignment. A queried
nonempty object subset turns the machine on iff it contains at least one
active object.

The exact hidden support is therefore the 255 nonzero eight-bit vectors and
the observation is noiseless OR group testing. Object color, shape, and
material are labels rather than causal semantics. A singleton query is nearly
balanced under the uniform prior and is both a myopic and depth-two optimum by
symmetry. The LLM's natural-language rule generation is not load-bearing.

Paper: `https://arxiv.org/abs/2402.06025`.

## CausalGame

The official source is pinned at
`34223510a0e7466ad399ff19714425f8708b7fbe`.

The 14 scenarios are variants of a few published SCM families, including
antenna trap, deployment-zone trap, and weather noise. Equations and
configuration are directly executable, the action is batch deployment, and
there is no native setup action that unlocks a later experimental action.
The LLM acts as an experiment proposer/explainer rather than owning an
open semantic support or likelihood.

The generator's NumPy route can be seeded, but Python's global `random` route
is not fully seeded in the release, so exact paired episode replay also needs
a wrapper.

Paper: `https://arxiv.org/abs/2607.04293`.
Code: `https://github.com/viewsetting/CausalGame`.

## CyBench And CTF-Dojo

CyBench is pinned at
`1097a7226eb034d3821208114da38f10b8627ab1`; CTF-Dojo is pinned at
`8ef064a93c73bc66f8681b5648948c43f9c9a8ba`.

CyBench has real machine-verifiable prerequisite execution: 156 subtasks
across 43 current metadata files, many with command, context, or solution-file
dependencies. But each CTF is a separate fixed challenge. The release does
not define alternative latent worlds, a prior over worlds, or
question-conditioned observations under a common hidden state. Failed
subtasks can also reveal their answers before advancing. CTF-Dojo supplies
many executable challenges but leaves the same missing Bayesian unit.

This is useful inspiration for a future gated semantic environment, not a
direct BED substrate.

CyBench paper: `https://arxiv.org/abs/2408.08926`.
CyBench code: `https://github.com/andyzorigin/cybench`.
CTF-Dojo paper: `https://openreview.net/forum?id=sQPeclxRBG`.

## Consequence

Do not spend OpenRouter credit on these releases now. The immediate
development priority remains the LongVid contrastive path-belief gate:
external semantic transitions and exact necessary-clip endpoints already
exist, a replicated first-action tradeoff is measured, and the LLM can be
made the sole carrier of the evolving open-world evidence-chain support.
