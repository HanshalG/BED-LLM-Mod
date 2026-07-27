# CUP Release Source Audit Preregistration

Date: 2026-07-28

## Motivation

The COLM 2026 paper *Uncertainty as a Planning Signal: Multi-Turn Decision Making for
Goal-Oriented Conversation* is directly aligned with the active claim. It formulates
conversation as sequential experimental design over a hidden target, uses an LLM to
propose natural-language actions, and reports a non-myopic MCTS method outperforming
BED-LLM on four recommendation datasets.

An official repository is publicly available:

- repository: `https://github.com/ninglab/CUP`;
- observed `main`/`HEAD`: `4695d0e236e430d48ee05b701e499708e36ac852`;
- commit date: `2026-05-29`;
- paper datasets: Beauty, Fashion, Home, and Inspired.

Only repository metadata and the paper have been inspected. No repository file, task
record, target, candidate, prompt, or trace has been opened.

## Admission Questions

The audit separates two claims:

1. **Executable non-myopic dialogue:** can the official release reproduce a paired
   long-horizon policy comparison against a myopic EIG policy?
2. **Irreducibly LLM-native BED:** does the LLM create or interpret semantic
   hypothesis/action/observation structure that is not already available as a complete
   finite table?

The first may qualify as supporting evidence. It is headline-admissible only if the
second also passes. Natural-language surface text by itself is insufficient.

## Access Boundary

The audit will:

1. clone only commit `4695d0e236e430d48ee05b701e499708e36ac852`;
2. record a tracked-file content digest and inspect licenses, README, schemas,
   dependencies, configs, and runtime code;
3. inspect task identifiers and metadata fields only;
4. freeze all available evaluation IDs before opening any dialogue, candidate
   description, target, attribute value, or recorded response.

For each paper domain, order unique IDs by
`SHA256("cup-24422:" + domain + ":" + task_id)`. Freeze:

- Beauty: `10` mechanics, `40` development, `100` confirmation, rest retained;
- Fashion: `10` mechanics, `40` development, `100` confirmation, rest retained;
- Home: `10` mechanics, `40` development, `100` confirmation, rest retained;
- Inspired: `10` mechanics, `30` development, `40` confirmation, rest retained.

If released counts differ from the paper or IDs cannot be separated from endpoint
values, fail and amend prospectively before content access. Do not substitute a
different dataset version or regenerate target pools.

## Zero-Call Source Gate

After the manifest is frozen, only mechanics values may be opened. The audit must
determine:

- whether the exact paper datasets and preprocessing artifacts are present;
- whether the paper's user simulator, belief update, action proposal, EIG, MCTS,
  commitment, and endpoint logic are executable;
- whether model names, prompts, sampling settings, search budget, and dependencies are
  specified;
- whether paired policies can share targets, candidate pools, simulator randomness,
  and turn limits;
- whether the myopic comparator is a genuine one-step objective with otherwise matched
  action generation and compute;
- whether candidate support is fixed or regenerated from dialogue;
- whether possible observations and candidate compatibility are produced semantically
  by an LLM or read from released attributes;
- whether the complete action/observation response table can be enumerated without an
  LLM after preprocessing;
- whether terminal success is computed from a hidden target outside the planner score.

Executable admission requires all runtime components and a paired myopic comparator.
Headline admission additionally requires at least one load-bearing LLM-native
mechanism:

- open-ended semantic action generation whose useful action space is not exhaustively
  released;
- semantic observation likelihood or compatibility inference not reducible to a
  provided attribute lookup;
- history-dependent hypothesis/support generation or transition dynamics.

If CUP's LLM only verbalizes actions over a complete finite candidate-attribute table,
record it as a potentially strong non-myopic supporting benchmark, not the headline.

## Conditional Experiment

No model call is authorized by this source audit. A full executable pass permits a
separate response-blind experiment preregistration with:

- one released domain selected by a task-value-blind structural rule;
- official CUP or an exact depth-two adaptation;
- a compute-matched one-step EIG control;
- a random/fixed-action control;
- naive thinking as a separate baseline only;
- paired targets and common simulator randomness;
- hidden-target success, turns, truth rank/log probability, and realized information
  gain as outcomes.

Serving must establish exact action/observation parsing and counterfactual consistency
before development. Development must show a positive paired endpoint gain and
truth-anchored ranking before confirmation.

There is no protected OpenRouter reserve. The full available balance may be allocated
by expected scientific value after validity gates. OpenRouter is the only model
backend; OatML, Slurm, and SSH are prohibited.
