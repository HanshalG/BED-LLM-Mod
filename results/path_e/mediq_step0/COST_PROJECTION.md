# MediQ Step 0 OpenRouter Cost Projection

Date: 2026-07-13

Frozen config: `configs/config_mediq_step0_smoke_openrouter.yaml`

## Run shape

- Official iMEDQA release at commit `faa2ce62fef0423e35af4c31d7537aad973173eb`.
- Raw-file SHA-256: `3bfc7090d060dd8d11e4237344ed78846707faab433a84d078191627ad3c9526`.
- Five usable cases at offset 0, two rounds, five candidates per round.
- One-step EIG only; Gemma 4 26B A4B without thinking for both roles.
- Trial batch size 5, OpenRouter concurrency 128, maximum output 1024 tokens.
- Three official rows without context or atomic facts (source IDs 224, 298, and 779)
  are explicitly excluded from the 1,269-case interactive subset and recorded in every
  run's data manifest.

## Logical requests

The zero-cost routing-model dry run over the same five official records made exactly
235 logical requests:

| Stage | Requests |
|---|---:|
| Initial finite-label prior | 5 |
| Root candidate generation | 10 |
| Option-conditioned outcome likelihoods | 200 |
| Fact-Select patient | 10 |
| Relevance/outcome mapping | 10 |
| **Total before structured retries** | **235** |

The largest concurrent stage contains 100 likelihood requests, so concurrency 128
already covers the stage; 256 would not shorten the critical path for this run.

## Dollar reservation

- Previous canonical Paprika smoke: 210,536 tokens, 669 requests, $0.04443824.
- Expected MediQ volume: approximately 180k-260k tokens plus bounded structured retries.
- Expected cost: approximately $0.04-$0.08.
- Conservative config reservation: **$0.12**.
- Ledger before launch: $16.71009983 spent of the user-authorized $40 cap;
  $23.28990017 remains.

The smoke is an environment and probabilistic-mechanics gate, not endpoint evidence.
Its five-case accuracy is diagnostic only. Automated passage still requires a separate
manual review of question atomicity, response-category exclusivity, and fact relevance.
