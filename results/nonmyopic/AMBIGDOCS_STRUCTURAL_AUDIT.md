# AmbigDocs Structural Audit For LLM-Native Sequential BED

## Source

- Dataset: `yoonsanglee/AmbigDocs`
- Hugging Face revision:
  `19d318a5e6717f63b9b864aa804bc57f69e824df`
- Development file SHA256:
  `43ab72b880337fac6442ea04accbda38d6a8e785eb2db4970585fac7ae775d68`
- Development rows: `3,610`
- Project page: `https://ambigdocs.github.io/`

The test split has not been downloaded or inspected in this audit.

## Support Depth

Number of disambiguated entity documents per development question:

| Documents | Rows |
|---:|---:|
| 2 | 1,790 |
| 3 | 982 |
| 4 | 480 |
| 5 | 223 |
| 6 | 88 |
| 7 | 36 |
| 8 | 10 |
| 9 | 1 |

There are 838 development rows with at least four candidate entities and 358 with at
least five. This is substantially deeper than CondAmbigQA-2K, where only seven of
2,000 rows have four or more candidate intents.

## BED Construction

For one AmbigDocs row:

- candidate hypotheses are its externally supplied entity documents;
- a hidden target entity is sampled uniformly by a frozen seed only after all policy
  scores freeze;
- the policy sees the ambiguous question and candidate document titles/text, but no
  target marker;
- the LLM generates free-form semantic binary clarification questions;
- separate LLM calls map candidate documents to `Y/N/U` likelihood labels;
- an independent responder sees only the hidden target document and generated
  question, producing the realized `Y/N/U` transition;
- posterior mass on the hidden entity after two questions is the endpoint.

The target cannot disappear because the support is fixed by the benchmark. The LLM
still does irreducible semantic work by generating the action space and the
question-to-document likelihood map. Branch-conditioned followup generation makes the
second action path-dependent.

## Controls

A future efficacy gate should compare, with shared generated calls and target:

- myopic receding EIG;
- depth-two semantic lookahead;
- compute-matched width search;
- seeded random root.

Both lookahead and width should consume equal question-generation and likelihood-call
budgets and execute exactly two independent responses.

## Development Boundary

Development row 49 (`qid=43608`, six entities sharing “Minsk”) was opened during this
audit and may be used for target-free serving development. No hidden target has been
sampled. All remaining qualifying dev rows are development inventory, not pristine
confirmation.

The official test split remains sealed for a separately preregistered confirmation.
No OpenRouter calls or OatML resources were used in this audit.

## Decision

AmbigDocs passes the zero-cost structural screen and replaces PSCon as the next
LLM-native route. Before any efficacy run, require a 10-call target-free serving gate
on the already-open six-entity row:

1. five one-line semantic binary questions;
2. five exact six-character `Y/N/U` likelihood maps.

Passage may authorize one separately frozen development efficacy run. Failure closes
the exact AmbigDocs interface before target sampling.
