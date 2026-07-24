# ClinDiag Filtered-Retention Recovery Gate Result

Date: 2026-07-24

Status: **failed; the three-batch ClinDiag recovery line is closed.**

## Result

The preregistered seed-`24300` run completed exactly 30 requests with zero reasoning,
structured retries, parser failures, or source-target leaks. Three independent
generate-filter batches were used for every original and replay path.

| Case | Old pruned | New introduced | Final/replay size | Exact Jaccard | Truth initial -> final/replay |
|---|---:|---:|---:|---:|---:|
| `27223150` | 0/12 | 0/0 | 12/12 | 1.00 | .76 -> .76/.76 |
| `rare130` | 11/12 | 10/10 | 11/11 | .10 | 0 -> 0/0 |

The substantive-transition and truth-stability gates passed. The cardinality gate
failed because the informative case produced only ten unique valid replacements across
three batches plus one retained old diagnosis.

## Inspection

For `27223150`, `lab_2` was a mild anemia result. Every old diagnosis remained above
threshold, so retaining the unchanged 12-item support was sensible.

For `rare130`, positive CMV IgM/IgG with low avidity strongly rejected 11 old
hypotheses. The three replacement batches nevertheless collapsed onto lexical variants
of maternal CMV infection, pneumonitis, myocarditis, and pericarditis. The original and
replay supports shared only two exact strings (Jaccard `.10`). Neither included fetal
cytomegalovirus syndrome, so truth score remained zero.

This is not a fourth-batch problem. Repeating the same prompt generated increasingly
redundant disease variants rather than broader semantic hypotheses. It also exposes a
ClinDiag target/evidence mismatch: the visible chart evidence describes maternal CMV
disease, while the benchmark target is a fetal syndrome not represented by the
selected stored observation.

The Listeria case similarly contained "listeriosis in pregnancy," which the strict
audit scored `.76` against "Listeria monocytogenes bacteremia." The benchmark's final
diagnosis granularity is not consistently aligned with the generic stored evidence
chunks.

## Decision

The exact three-batch ClinDiag recovery line is closed. Do not add a fourth batch,
loosen the `.20` filter, lower the cardinality requirement, or retune these prompts on
the observed cases. Filtered retention can produce coherent support transitions, but
this fixed-slot ClinDiag construction does not provide a reliable target-aligned belief
space for non-myopic policy evaluation.

No semantic-likelihood, ranking-fidelity, or policy run is authorized from this result.
The next route should use an environment whose semantic target and observations are
explicitly aligned, and should encourage diversity across semantic categories rather
than repeated same-prompt lexical variants.

## Cost

- 30 requests: 14 GPT-5.4 and 16 GPT-5.4 Mini;
- 10,627 prompt and 11,245 completion tokens;
- zero reasoning tokens;
- `$0.07958150`;
- conservative project-ledger remainder: `$21.70577372`.
