# OpenRouter Model And Daily Budget Decision

Date: 2026-08-06 (Europe/London)

## Hard Envelope

- Daily account-wide cap: `$5.00`; unused allowance expires at London midnight.
- Every paid block requires a dated ledger initialized from authenticated cumulative
  usage and a preregistered run cap before adapter construction.
- Observed credits/usage at review: `$245.000000000 / $217.297890263`, leaving
  `$27.702109737`.
- The newly reported `$30` top-up is not yet visible in the authenticated credits
  endpoint. It is not counted until it posts. The existing observed balance is still
  sufficient for the next five full daily caps.
- No borrowing across days and no reserve. A day may finish below `$5` when the next
  scientific gate is not ready; calls are never added merely to consume allowance.

## Current Intelligence-Cost Screen

Prices below are the live OpenRouter catalog values at review time.

| Model | Input/output per 1M | External screen | Direct Number Game evidence | Decision |
| --- | ---: | --- | --- | --- |
| GPT-5.6 Luna | `$0.10 / $0.60` | AA Intelligence Index `51` at max reasoning | Exact-10 passed cleanly for `$0.003578`; unchanged scale interface later recorded `15/1,200` forced-length responses and failed closed on strict JSON before an endpoint | Best repair candidate after the mandatory Qwen control; not yet a formal planner replacement |
| DeepSeek V4 Flash 0731 | `$0.09 / $0.18` | AA Intelligence Index `50` at max reasoning | Exact-10 transport and top-level JSON were clean, but one conditioned draw produced `0/24` valid executable rules | Cheapest intelligence point; require a conditioned-support reliability gate before efficacy |
| Qwen3.7 Plus | `$0.32 / $1.28` | lower generic price-performance than the two new candidates | Repeated task-specific non-myopic effects, two mechanism cohorts, and a fully fresh source stage with exact mechanics | Retain for the paper-critical pending control |

Generic scores nominate Luna and DeepSeek, but they do not test the irreducible task
here: producing diverse, history-consistent, executable semantic hypotheses under a
strict repeated interface. The direct task evidence therefore outranks a one-point AA
difference. Luna's observed failure is scale/interface reliability; DeepSeek's is
conditioned semantic support quality.

## Six Daily Blocks

| London day | Maximum | Authorized use |
| --- | ---: | --- |
| Aug 6 | `$5.00` | **Closed.** Fully fresh Qwen source used `$4.26043712`; remaining `$0.73956288` expires. |
| Aug 7 | `$5.00` | Mandatory hash-bound Qwen history-blind control: exact 3,072 accepted calls, expected about `$3.21`, hard cap `$4.25`. Run regardless of source science because mechanics and changed-root authorization passed. No competing paid block before it is banked. |
| Aug 8 | `$5.00` | Freeze and run a small matched reliability gate for Luna and DeepSeek 0731. Test conditioned support floors, strict schema completion, forced-length rate, and a single preregistered format-only retry path. Cap the screen at `$0.20`. Only a passing model may use the rest of the day for a paired downstream efficacy gate. |
| Aug 9 | `$5.00` | If Aug 8 efficacy passes, run an independent fresh model-family replication. If it fails, spend only on the failure-specific first-link diagnosis; do not rerun the same endpoint. |
| Aug 10 | `$5.00` | Test whether model support-quality improvement predicts realized non-myopic policy advantage on fresh histories, using paired roots and external canonical targets. This is the missing link in the current Qwen result. |
| Aug 11 | `$5.00` | Cross-model confirmation with the best task-qualified budget model, conditional on prior gates. If no model qualifies, redirect to a newly frozen LLM-native environment/mechanism test rather than another Number Game scale rerun. |

Each day's exact block remains conditional on prior results. A new dated ledger must be
created from live usage at the start of each London day; this table is an allocation,
not preauthorization to spend future-day balances.

## Model Order

1. Finish the paper-critical Qwen control without changing models.
2. Give Luna the first repaired-interface gate because it passed conditioned support
   quality but produced a measurable forced-length tail at scale. The gate must bound
   that rate; merely increasing the output limit is not enough.
3. Keep DeepSeek 0731 as the cheapest challenger, but require direct evidence that its
   zero-valid conditioned draw is controlled before any large run.
4. Promote a replacement only after both strict mechanics and paired non-myopic
   efficacy pass. Token price alone cannot replace the model doing the scientific work.

External references:

- OpenRouter GPT-5.6 Luna: https://openrouter.ai/openai/gpt-5.6-luna
- OpenRouter DeepSeek V4 Flash 0731:
  https://openrouter.ai/deepseek/deepseek-v4-flash-0731
- Artificial Analysis GPT-5.6 Luna max:
  https://artificialanalysis.ai/models/gpt-5-6-luna/
- Artificial Analysis DeepSeek V4 Flash 0731 max comparison:
  https://artificialanalysis.ai/models/comparisons/deepseek-v4-flash-vs-gpt-5
