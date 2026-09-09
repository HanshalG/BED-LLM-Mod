# Author string bank: development source gate passes

The author-maintained [ILP dataset collection](https://huggingface.co/datasets/andrewcropper/ilp-datasets/tree/33e166058404fd5eec3ec0f5080df9befbe884a8)
declares MIT in its dataset-card metadata and describes 329 character-level string
tasks. Its strings README attributes the source to Cropper's Playgol paper. This
is a larger collection than the original paper's 94 tasks, not a demonstrated
byte-identical reproduction of that historical release.

Only strings/1/train and strings/10/train were opened, as prospectively designated
in AUTHOR_STRING_SOURCE_DEV_SCOPE_20260909.md. No other task contents were fetched.
These two tasks are permanently development data; other tasks require overlap checks
against the previously parsed SYNTRA derivative before being called fresh.

Revision: 33e166058404fd5eec3ec0f5080df9befbe884a8. Hashes of all six downloaded files
are banked in AUTHOR_STRINGS_DEV_AUDIT_20260909.json. Full collection size is roughly
432GB according to the card; no collection clone/download was attempted. Files were
fetched individually under a one-megabyte per-file cap.

## Measured contract

| Development task | Distinct inputs | Positive output facts | Negative facts | Label conflicts |
|---|---:|---:|---:|---:|
| 1 | 10 | 47 | 799 | 0 |
| 10 | 10 | 114 | 3762 | 0 |

Each input ID has input characters, a matching declared input width, and positive
output characters. Positions are contiguous from one in both input and output.
The strict ground-fact grammar uses Lark, executes no Prolog, rejects directives,
and does not evaluate source or generated code. Five synthetic tests (.14s) cover
parsing, escaped quotes, contradictory labels, positional gaps, width mismatch and
missing output records; lint passes.

Ten distinct inputs permit one initial observation, six candidate queries and three
disjoint targets with a three-query budget. This clears the derivative release's
cardinality obstruction. It does not establish any non-myopic gap or model quality,
and the count has been verified only for the two designated tasks, not all 329.

Positive output facts can be reconstructed into finite observed strings on these
tasks. There is no verified evaluator for arbitrary new inputs. Restrict any future
experiment to a frozen table of provided examples; do not let an LLM fabricate labels
for invented queries. The current parser rejects unresolved empty-output cases and
is not a full Prolog implementation. Such a rejection on another task must be banked,
not silently treated as an empty string or dropped from evaluation.

## Next zero-call experiment

On these opened development tasks, define one shared symbolic string-program search
space with a declared prior, bounded executable compiler, and uniform search budget.
Use only the chosen initial observation to construct compatible hypotheses. Freeze
input-index roles before the numerical run. Compute ordinary receding h1/h2/h3 and
full-budget terminal predictive risk under the exact finite reference, with identical
queries/budget/targets, random and open-loop controls, and work/coverage reporting.

Do not select tasks, priors or query roles after observing a gap, truncate a reference
and call it complete, or confuse a finite reference's internal risk with real-data
calibration. A null should stop that candidate allocation. A classical gap is only
supporting evidence; it must be followed by a separately frozen held-out LLM proposal
and joint conditional prediction gate. Real table outcomes must evaluate the policy,
not its own confidence. The actual goal remains LLM-native sequential non-myopic BED.

No model calls, predictions scored, policy comparison, or paid authority in this
source gate. The $4.23832314 remaining London allowance is unchanged. Previous turn
ruled out an undersized derivative; this turn locates and verifies a usable larger
development source. Research goal active and incomplete.
