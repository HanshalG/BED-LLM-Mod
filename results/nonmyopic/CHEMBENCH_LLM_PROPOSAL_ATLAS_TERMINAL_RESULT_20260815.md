# ChemBench LLM Proposal-Atlas Terminal Result

Date: 2026-08-15 (Europe/London)

## Disposition

The exact primitive-pool proposal-atlas interface failed closed and authorizes
nothing. It must not be rerun with new seeds, relaxed gates, or repaired
responses. No policy endpoint or v5 outcome opened.

The run made 126 zero-retry requests to exact
`deepseek/deepseek-v4-flash-0731`, explicitly nonreasoning. It cost
`$0.0278822622`; authenticated cumulative usage closed at `$220.376693994`.

## Transport

- Exact requested/returned model: 126/126.
- HTTP attempts and accepted requests: 126/126, with zero retries.
- Clean terminal responses: 107/126 (`84.92%`, below the frozen 90% gate).
- Finish reasons: 107 `stop`, 18 `error`, and one `length`.
- The 18 error-finish contents were corrupted by repeated image-control tokens.
- Reasoning tokens: zero; one adapter forced-exit event was recorded.

Transport is therefore a gate failure, but it is not the main scientific
failure: the clean responses also miss the semantic requirements broadly.

## Proposal Semantics

| Metric | Residual-aware | History-blind |
|---|---:|---:|
| Clean rate | 84.13% | 85.71% |
| Strict schema rate | 84.13% | 79.37% |
| Item compile rate | 31.35% | 36.11% |
| Four-item executable response rate | 3.17% | 1.59% |
| Truth recall at four | 3.17% | 6.35% |
| Core-family recall at four | 12.70% | 14.29% |
| Mean modifier F1 | 20.37% | 23.02% |

Among clean residual-aware responses alone, schema validity is 100%, but item
compile rate is only 37.26%, four-item executable rate and truth recall are
both 3.77%, core-family recall is 15.09%, and modifier F1 is 24.21%. Thus the
semantic null is not explained by transport failures.

Paired semantic differences are 10 wins, 40 ties, and 13 losses for the
residual-aware arm. On the nine held-out tasks, fresh residual-aware proposals
recover one truth and one core family; the atlas recovers one truth and four
core families. Mean one-step risks are:

- source oracle: `0.0620104`;
- fresh residual-aware: `0.6127136`;
- atlas: `0.7049844`;
- history-blind: `0.7457816`;
- random typed edits: `0.7777391`.

Fresh and atlas proposals beat blind and random risk, but remain roughly ten
times the oracle risk and fail the frozen oracle-relative gates.

## Failure Anatomy

Across schema-valid responses, invalid compiled items are dominated by:

- 143 signatures absent from the fixed executable registry;
- 60 declared operations inconsistent with the declared parent;
- 39 candidates that merely reproduce represented or previously tried models.

The model heavily overproduces Michaelis-Menten, ordered-Bi-Bi, and mixed
inhibition candidates and shows no positive residual-aware advantage. The
categorical residual report is therefore not a usable semantic proposal signal
for this model/interface. Separately, allowing a public mechanism grammar but
accepting only hidden benchmark registry combinations is a poor approximation
to MDA's genuinely compositional compiler.

## Consequence

Do not build the recursive atlas or run d1/d2/d3 from this interface. A new
prospective route must change the scientific interface, not tune this result:

1. expose a finite public vocabulary of atomic edits instead of requiring a
   redundant full candidate signature and parent-relative operation;
2. compile every scientifically compatible composition, rather than only the
   benchmark's active registry combinations;
3. use compact task-anchored residual fingerprints or paired diagnostic
   contrasts that make the missing factor identifiable;
4. test standard compositional modifiers before obscure novel core families;
5. require residual-aware advantage and recursive expanded-pool fidelity
   before any policy endpoint.

## Bound Artifacts

- Daily result SHA256:
  `8c5d579cf875392143ad40d4141985d4bf7785a392ce370aa0873d1e36545a67`.
- Raw response bank SHA256:
  `9b1662179398b21b9d91b521d3be1099f895d314638c839132973b5ff2e615f2`.
- Semantic evaluation SHA256:
  `8d727786d187b8c3a542f2f99bfd8addb8ecab8bcab3ae437846dab91efea56d`.
- Independent semantic verification SHA256:
  `c029e8016bcf6f338b6d0f5054f33abad1fee9f6cbb615c071735a6f35290451`.
- Daily ledger SHA256:
  `a226f98ff6775fc8f8b91449ab12f3b08c0167e52e4f353c5fe73a4760fd4467`.
