# LLM-Native Preference Benchmark Source Screen

Audited: 2026-07-28. This is an exploratory zero-call release and control-flow
screen, not a preregistered efficacy experiment.

## Decision Criterion

A headline candidate must simultaneously provide:

1. hidden semantic state that the policy cannot replace with a released finite
   table;
2. a runnable counterfactual interaction model;
3. uncertainty that survives the initial interaction;
4. a scarce sequential horizon in which an earlier information action can
   improve later acquisition or execution; and
5. an endpoint external to the policy's own belief score.

## ATRBench

- Paper: `https://arxiv.org/abs/2605.28108`
- arXiv source archive SHA256:
  `e3d10d16cd2da4c6525ee92ad0dc9f399be9eec3e8cb28f7af3af05ae5dc5190`

ATRBench is the strongest conceptual match. Its Ask-to-Remember episodes make
the agent ask in a learning session for a standing rule that is unnecessary
now but useful in a later, uncertain test session. The paper describes 20
personas, 284 standing rules, 568 learning sessions, six domains, 74 tools, a
hidden-rule router, canonical user answers, and deterministic test-action
checks.

The release is not runnable as of the audit. The arXiv source contains no code,
dataset, artifact, or repository URL. Exact-title and benchmark-name searches
find no official GitHub repository, and the Hugging Face dataset API finds no
matching release. Reconstructing hidden rules, sessions, router labels, and
tool checks from prose would create a new benchmark rather than evaluate
ATRBench. Keep it on the release-watch list.

## ADAPT

- Paper: `https://arxiv.org/abs/2504.04040`
- Accepted version: `https://openreview.net/forum?id=Z8vtD1egtI`

ADAPT also has the right scientific shape: open-ended preference questions are
interleaved with long-horizon household execution, and manually defined
preference verifiers score the final trajectory. The accepted paper still
states that code will be released on acceptance, but exact-title,
Reflection-DPO, author, GitHub, and Hugging Face searches find no official
runtime or dataset. It is therefore unavailable rather than scientifically
rejected.

## PrefDisco

- Paper: `https://arxiv.org/abs/2510.00177`
- GitHub: `https://github.com/stellalisy/PrefDisco`
- Dataset collection:
  `https://huggingface.co/collections/stellalisy/personalized-reasoning`
- Audited dataset revision for `personalized_math`:
  `afe077a70471a6ddf69fe665fa944760ff1cbf85`

PrefDisco is the strongest conditional next route. It exposes sparse,
task-specific semantic preferences, limits discovery to five conversational
turns, and evaluates a personalized reasoning answer with instance-specific
rubrics. This creates native information scarcity and keeps the LLM's semantic
questioning and adaptation load-bearing.

The official GitHub repository is empty. Nine task datasets are listed on
Hugging Face, but they are auto-gated and require a user to agree to share
contact information. This workspace has no Hugging Face access token, so no row
or hidden preference was opened. Access must be granted explicitly before a
source/schema audit; the benchmark must not be reconstructed from the paper.

## PrefBench

- Paper: `https://arxiv.org/abs/2605.22855`
- Repository: `https://github.com/ChaosTheProducer/PrefBench`
- Commit: `d88fe85071196e89b777ec5433e8d10de9aa0a4c`
- Tree: `9373f9854f1fd883db097cc4f057e17708a494a9`

PrefBench is complete and reproducible, but it is not a headline candidate for
this project. Public test rows contain the full hidden buyer profile:
reservation value, price sensitivity, feature weights, patience, counter
strength, walkaway tendency, obscurity, loyalty, and impulsivity. The released
simulator maps those finite numeric fields to willingness to pay and
accept/counter/walkaway probabilities. The policy action is only a numeric
offer, accept, or walkaway.

An exact classical POMDP can integrate this released state and transition
model. Language may help emit an offer, but it does not own the hypothesis
space, likelihood, or path-dependent belief dynamics. PrefBench is
supporting-only.

## ClarQ-LLM

- Repository: `https://github.com/ygan/ClarQ-LLM`
- Commit: `e3bc7c80173f01d0a04fa42f6b24efd9266fd081`
- Tree: `5ce4ac485674a8b035501866bd67b4a75ee8a77e`

The English release contains 31 task types and 310 paired tasks. Each task has
2--6 hidden information nodes after its initial broad reply. Exact source
inspection found 144/310 tasks with a dependency edge, including 18 depth-three
and two depth-four trees. The provider keeps only root facts initially
available; serving a parent removes it and promotes its children. A vague
question is deflected, and the normal provider serves at most one semantically
matched fact per turn.

These mechanics are genuinely path-dependent, but the native horizon removes
the decision tradeoff. The loop permits about 11 follow-up seeker questions,
whereas every task has at most six hidden facts. Each successful targeted
question gains one fact, there is no query cost in task success, and a receding
myopic policy can collect every parent and child with at least five turns of
slack. Dependencies constrain order but do not make an enabling root sacrifice
immediate value or improve a scarce endpoint. Tightening the query budget would
manufacture the non-myopia rather than use the released environment. No paid
ClarQ-LLM experiment is authorized.

## ProactiveBench

- Paper: `https://arxiv.org/abs/2603.19466`
- Repository: `https://github.com/tdemin16/ProactiveBench`

ProactiveBench is multimodal and semantically native, but each item is a
multiple-choice choice between an answer, abstention, and one of a few user
interventions. The intervention reveals a better image and is followed by an
answer. It evaluates whether to ask for help, not adaptive experimental design
over multiple information actions. It is not the next sequential BED route.

## Next Action

PrefDisco is first in the runnable-candidate queue once its gated datasets are
explicitly accessible. The first stage must be a zero-call source/schema audit
that verifies:

1. hidden preference values and task rubrics are released without leaking into
   policy-visible prompts;
2. user answers are reproducible from released state or an official simulator;
3. the five-turn limit is enforced;
4. task success remains externally verifiable;
5. multiple candidate questions have answer-dependent continuation value; and
6. a fixed-support classical table cannot replace the semantic interaction.

ATRBench and ADAPT stay on the release-watch list. PrefBench and ClarQ-LLM are
closed as headline routes for structural, not cost, reasons.

## Accounting

- OpenRouter requests: `0`
- OpenRouter cost: `$0`
- OatML, Slurm, SSH, or cluster use: `0`
- Authenticated OpenRouter balance after the screen: `$8.990154844`
- Reserve: none
