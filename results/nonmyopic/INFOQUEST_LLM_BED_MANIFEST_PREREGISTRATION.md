# InfoQuest LLM-BED Manifest Preregistration

Frozen before inspecting any InfoQuest scenario content beyond mechanics IDs
`0` and `1`.

## Question

Does the released InfoQuest benchmark provide a sufficiently large,
source-grounded substrate for testing non-myopic experimental design over an
LLM's path-dependent semantic belief state?

InfoQuest is not yet policy evidence. This first gate only pins the source,
validates cross-file structure, and creates content-blind splits. It makes no
OpenRouter calls and does not inspect released trajectory outcomes.

## Frozen Source

- Hugging Face dataset: `bryanlincoln/infoquest`
- revision: `1f54a770a8bed73edcab86411254fff64ed53d25`
- required files: `seed_messages.jsonl`, `settings.jsonl`, `traits.jsonl`
- expected records: exactly `500`, with IDs exactly `0..499` in order

The three byte-level source hashes are frozen in
`scripts/infoquest_llm_bed_manifest.py`.

Each source record must match its exact released schema. Each of the two hidden
settings must have nonempty description, goal, obstacle, solution, and persona
fields; exactly five nonempty constraints; and exactly five nonempty checklist
items. Setting personas and seed messages must match the corresponding seed
record. Trait records must be aligned by ID and nonempty.

## Access Boundary

Mechanics IDs `0` and `1` were disclosed during source-shape inspection before
this preregistration. No other semantic content may be inspected while
building the manifest.

The public manifest may emit:

- source repository, revision, filenames, counts, and byte hashes;
- record IDs and SHA-256 hashes of canonical records;
- split IDs, sizes, and ordered hashes;
- structural gate booleans and zero-call accounting.

It must not emit seed messages, personas, hidden settings, traits, constraints,
solutions, or checklist text.

## Frozen Split

Use seed `24416`.

1. Mechanics is fixed to IDs `[0, 1]`.
2. Shuffle IDs `2..499` once with Python `random.Random(24416)`.
3. Opportunity is the first `80`.
4. Development is the next `30`.
5. Holdout is the remaining `388`.

Frozen ordered split hashes:

| Split | Records | SHA-256 |
| --- | ---: | --- |
| Mechanics | 2 | `463f2998327eb3a694145e6014444480b2235be84aa6cfd57871cc64f1cd816c` |
| Opportunity | 80 | `737296c449680cfb7c78aa03d44508af487eb9624d0e416d37cf5ddbaa01958b` |
| Development | 30 | `068587d494c71d5488da4f1d53ebe34be713c178083f6be98733085337c48b48` |
| Holdout | 388 | `91132054ecf67049deaf1644d8e6564f4c1538559a4735c676deafc69af6ea91` |

Combined split hash:
`1d8f5adbfd30677311ec1a150e2b7c1f7d0804c6ed1d9facf7d13d82b0957edf`.

## Conjunctive Gates

1. checkout revision matches exactly;
2. all three source hashes match exactly;
3. all three files contain exactly 500 aligned records;
4. IDs are exactly `0..499` in order;
5. every record passes the exact schema and cross-file checks;
6. every setting has exactly five constraints and five checklist items;
7. all four splits are nonempty, disjoint, and cover all 500 IDs;
8. the manifest contains no semantic source content;
9. OpenRouter calls, OpenRouter cost, and OatML jobs are all zero.

A pass authorizes only a separately preregistered zero-cost audit of the
official cached baseline trajectories on the opportunity split. Those
trajectories may establish structural opportunity and path dependence, but
cannot establish causal policy efficacy because they reveal only realized
paths. Development and holdout semantics remain sealed.

A failure closes this exact source/split line without threshold, record-count,
schema, or split repair.
