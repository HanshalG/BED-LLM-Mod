# OpenRouter Daily Budget And Model Decision

Date: 2026-08-07 (Europe/London)

## Live Account Boundary

Authenticated OpenRouter credits at review time report:

- total credits: `$245.000000000`;
- cumulative usage: `$220.113606154`;
- available balance: `$24.886393846`.

The user-reported additional `$30` is not yet present in the authenticated
credit total and is therefore not spendable in the project ledger. It will be
counted automatically only after the provider reports it.

The account-wide cap remains exactly `$5.00` per Europe/London calendar day.
Each paid day starts from authenticated cumulative usage. Posted spend,
locally measured accepted-request spend, and precharged in-flight exposure are
reconciled conservatively. Calls cannot borrow from another day, use unposted
credit, or bypass the cap through concurrency. Unused headroom does not roll
over and is not a reason to make scientifically unmotivated calls.

On August 7 the immutable ledger records `$2.815715891` of useful paid work and
`$2.184284109` remaining. No dependency-valid paid endpoint remains today, so
the remainder is deliberately unspent rather than used to reopen a null or
weaken a gate.

## Live Model Frontier

The live OpenRouter catalog reports:

| Model | Input / output per 1M | Modalities | Role |
|---|---:|---|---|
| GPT-5.6 Luna (`openai/gpt-5.6-luna`) | `$0.10 / $0.60` | text, image, file | Multimodal belief model and medium-reasoning naive baseline |
| DeepSeek V4 Flash 0731 (`deepseek/deepseek-v4-flash-0731`) | `$0.09 / $0.18` | text only | High-volume nonreasoning text support and planning |

Generic intelligence-per-dollar screens make both models attractive. They do
not measure the repeated task interface needed here: history-consistent,
diverse, executable semantic hypotheses under strict output and support
constraints. Direct project evidence therefore controls promotion.

Artificial Analysis now reports independent max-reasoning rows for both
current checkpoints: Luna scores `52.32` and DeepSeek V4 Flash 0731 scores
`51.77`. Their weighted Intelligence Index costs are about `$0.0471` and
`$0.0271` per task respectively, so 0731 is about 42% cheaper on that generic
reasoning screen. The same dataset reports output speeds near `190.4` and
`104.2` tokens/s. OpenRouter's authenticated catalog is cheaper still: Luna
`$0.10/$0.60` and 0731 `$0.09/$0.18` per million input/output tokens.
DeepSeek is therefore the text-only throughput default, while Luna is reserved
for images or the separately labelled reasoning baseline. Direct interface
gates still control every promotion because the generic max-reasoning screen
does not measure nonreasoning support validity.

## Direct Task Evidence

| Model / interface | Cost | Strict mechanics | Conditioned support | Decision |
|---|---:|---|---|---|
| Luna exact-10 | `$0.003577600` | pass | 123 valid conditioned hypotheses | Best observed cost per usable conditioned hypothesis |
| DeepSeek 0731 exact-10 | `$0.003258592` | pass | 92 valid; one zero-valid draw | Cheaper tokens, weaker usable support |
| Luna reliability128 | `$0.057383000` | 4 malformed/forced | min/mean `0 / 15.925`; 3/120 below 4 | Gated null |
| DeepSeek reliability128 | `$0.034477701` | 128/128 clean | min/mean `0 / 12.108`; 20/120 below 4 | Gated null |
| DeepSeek diversity-V2 | `$0.033681750` | 128/128 clean | mean `8.625`; 18/120 below 4 | Gated null |

Luna remains the correct frozen Bongard model because it accepts images and
has the stronger observed usable-support economics. Its long-tail strict
completion risk is handled by the already frozen serving and terminal-validity
gates, not by silent retries or substitution.

DeepSeek 0731 is not promoted as the nonreasoning Number Game support model:
both fresh conditioned-support screens failed. It remains the first cheap
candidate for a substantively new text-only interface and is the frozen
RegretBench support/planning model. Luna medium reasoning is the explicitly
labelled naive-thinking baseline. Reasoning scores must not be used to choose
the nonreasoning planner or environment model.

## Daily Allocation Rule

- August 8: run the exact dependency chain with one account opening: Bongard
  Luna naive smoke `$0.20`; RegretBench DeepSeek support smoke/development
  `$0.20/$0.50`; enriched policy smoke `$0.20`; Luna naive-thinking smoke
  `$0.20`; and mixed-model policy development `$3.50`. Maximum is `$4.80`,
  leaving `$0.20` safety headroom. Every descendant remains pass-gated and no
  block may be added merely to consume the headroom.
- August 9: use only a separately preregistered dependency-valid block. Failed
  Number Game descendants stay closed.
- August 10 has one mutually exclusive branch. If the Aug 9 SMC policy banks a
  literal independently verified pass and its frozen report requires
  confirmation, run only the hash-bound RegretBench SMC confirmation: `$0.20`
  parent bank plus `$3.50` policy, maximum `$3.70`. Otherwise run only the
  frozen Luna Bongard serving and mechanics wrapper, component cap `$2.00`.
  Opening either branch forbids the other for that day.
- August 11--14: reserve up to `$4.75` for the exact frozen Bongard
  development blocks A--D, one block per day, conditional on every predecessor
  and read-only preflight passing. After each clean main block, run the bound
  eight-request Luna reasoning baseline with a `$0.20` cap. Total daily
  worst-case exposure is `$4.95`.

The target is useful spend near `$5` on an active paid day, not nominal spend.
A day closes below the cap when no scientifically admissible block is ready.
