# CUP Release Source Audit Result

Date: 2026-07-28

**Status: failed before dataset-value access and before any model call.**

## Pinned Source

- official repository: `https://github.com/ninglab/CUP`;
- commit: `4695d0e236e430d48ee05b701e499708e36ac852`;
- tracked files: `16`;
- tracked-content manifest SHA256:
  `d8c6f3349d99edd8acd250457ad9aa4cd86555af5c3770327c2c865e4e5fc3b5`;
- checkout state: clean.

The repository contains Python source and prompt documentation only. It contains no
dataset, candidate cache, embedding cache, result artifact, dependency lock, or
license.

## Reproducibility Failure

The runtime expects unversioned local directories `./Inspired` and `./lavic`. The
README links their upstream repositories but does not pin commits, checksums, download
commands, or preprocessing versions. Therefore:

- no official evaluation ID is present to freeze;
- no preregistered metadata-only domain partition can be materialized;
- the paper's exact 300-candidate pools are unavailable;
- no released baseline implementation reproduces BED-LLM, myopic EIG, UoT, MISQ-HF,
  or the other reported comparisons.

Candidate pools are regenerated locally with SBERT. If a target is absent, both
dataset loaders replace a randomly chosen retrieved candidate using
`random.randint(...)`. The repository sets no Python, NumPy, Torch, or Transformers
seed. Consequently, even an independently reconstructed dataset would not recover the
paper's exact candidate pools.

The paper specifies commitment threshold `theta=0.8`, while the released CLI defaults
to `theta=0.6`. This is another unresolved paper/runtime mismatch.

## Hidden-Target Leakage

The released non-myopic planner is not target-blind:

1. `run.evaluate(...)` reads the ground-truth target item.
2. It constructs `MCTS(env, target_item, ...)`.
3. Every MCTS expansion and rollout calls `_temp_env(...)`.
4. `_temp_env(...)` installs `DeterministicUserSimulator(self.target_item)`.
5. That simulator answers questions and recommendations from the actual hidden target.

Thus MCTS evaluates every counterfactual trajectory under the true test target. It
does not sample a hypothetical target from the current belief, enumerate expectations
under belief mass, or carry particles. Two tasks with identical observable history and
belief can receive different actions solely because their hidden target objects differ.

A direct source-level toy reproduced this. With the same uniform belief over five
items, the same candidate order, the same action set, and Python seed `24422`, changing
only the private target produced:

```text
target A -> rec_A  visits {'rec_A': 169, 'ask_x': 165, 'ask_y': 165}
target B -> ask_y  visits {'rec_A': 165, 'ask_y': 169, 'ask_x': 165}
```

For target A, the planner knows that committing to the insertion-order top candidate
will succeed. A valid policy conditioned on the identical belief state cannot have
that information. The released CUP results therefore cannot serve as evidence that
non-myopic planning under uncertainty beats a myopic policy.

## LLM Role Audit

The released environment is also not irreducibly LLM-native:

- every ask action is one of six to nine fixed attributes;
- the LLM is instructed to emit every available attribute;
- proposed options are accepted only when they exactly match finite stored values;
- omitted attributes and options are filled by a structural fallback;
- inner MCTS nodes and rollouts use only structural actions;
- candidate compatibility and EIG come from a complete item-by-attribute table;
- the user simulator computes the selected option deterministically from the target
  attribute before calling the LLM;
- the environment updates on that structured option, not on the generated language;
- the LLM-generated user response and question wording are surface realizations;
- candidate support is fixed and only shrinks.

The repository's prompt documentation explicitly says the LLM only verbalizes the
precomputed ground-truth preference. Removing both LLM verbalizers leaves the planner's
structured trajectory, candidate filtering, reward, and success unchanged.

## Gate Decision

CUP fails both preregistered admission questions:

- the exact paper experiment is not reproducible from the official release;
- the non-myopic planner directly accesses the hidden target;
- no compute-matched myopic comparator is released;
- the LLM does not own the action, observation, likelihood, or support dynamics.

No task value or endpoint was opened, no upstream dataset was substituted, and no
OpenRouter call was made. Correcting the target leakage could make CUP a useful
classical finite-support benchmark, but it would be a new experiment and would not
repair the missing LLM-native mechanism. Do not allocate the current API budget to
this route.

OpenRouter spend: `$0`.
