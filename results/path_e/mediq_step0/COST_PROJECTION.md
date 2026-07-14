# MediQ Step 0 OpenRouter Cost Projection

Date: 2026-07-14

Frozen config: `configs/config_mediq_step0_smoke_openrouter.yaml`

## Run shape

- Official iMEDQA release at commit `faa2ce62fef0423e35af4c31d7537aad973173eb`.
- Raw-file SHA-256: `3bfc7090d060dd8d11e4237344ed78846707faab433a84d078191627ad3c9526`.
- Five usable cases at offset 0, two rounds, five candidates per round.
- Candidate actions are atomic yes/no predicates with the fixed outcome support
  `Yes`, `No`, and `Information unavailable / not in record`; validated candidates are
  retained while only rejected deficits are regenerated. The action contract also
  rejects target-decode/management queries, derived clinical summaries, and semantic
  repeats of prior or already accepted predicates.
- One-step EIG only; Gemma 4 26B A4B without thinking for both roles.
- Trial batch size 5, OpenRouter concurrency 128, maximum output 1024 tokens.
- Three official rows without context or atomic facts (source IDs 224, 298, and 779)
  are explicitly excluded from the 1,269-case interactive subset and recorded in every
  run's data manifest.

## Logical requests

After the complete answer-space, action-contract, and set-dedup repair, the zero-cost
routing-model dry run over the same five official records made exactly 305 logical
requests:

| Stage | Requests |
|---|---:|
| Initial finite-label prior | 5 |
| Root candidate generation | 10 |
| Temperature-zero semantic candidate validation | 50 |
| Temperature-zero candidate-set deduplication | 10 |
| Option-conditioned outcome likelihoods | 200 |
| Fact-Select patient | 10 |
| Category-blind explicit-relevance judgment | 10 |
| Outcome mapping | 10 |
| **Total before bounded repairs** | **305** |

The largest concurrent stage contains 100 likelihood requests, so concurrency 128
already covers the stage; 256 would not shorten the critical path for this run.

## Dollar reservation

- Initial MediQ smoke: 88,243 tokens, 235 requests, $0.01416921.
- Expected repaired volume: approximately 105k-150k tokens plus bounded repairs.
- Expected repaired cost: approximately $0.02-$0.05.
- Conservative config reservation: **$0.12**.
- Ledger before the next exact repeat: $16.77506738 spent of the user-authorized $40
  cap; $23.22493262 remains. Three failed-closed repair attempts and three complete
  diagnostic/manual-fail smokes are recorded in `EXPERIMENTS.md`.

The smoke is an environment and probabilistic-mechanics gate, not endpoint evidence.
Its five-case accuracy is diagnostic only. Automated passage still requires a separate
manual review of question atomicity, response-category exclusivity, and fact relevance.
