# OpenRouter Five-Dollar Daily Plan

Date frozen: 2026-08-06

## Account And Rule

Authenticated state at freeze:

- total credits: `$245.000000`;
- total usage: `$217.297890`;
- available balance: `$27.702110`.

The newly reported `$30` top-up is still not visible on this API key as of
`2026-08-06 02:11 BST`, so it is not counted in the executable balance until
the credits endpoint posts it. If it posts in full without intervening usage,
the balance becomes `$57.702110`, enough for eleven full `$5` days plus
`$2.702110`.

The hard rule is `$5.00` per Europe/London calendar day, account-wide, with no
rollover. The operating target after August 6 is `$4.75--$5.00` of useful
work per day, not merely a `$5` maximum. Every paid runner checks cumulative
live usage against a frozen opening baseline before its first provider call,
and checkpoints locally measured cost because provider usage can post late.
Concurrency changes throughput, not the spend ceiling.

After the primary daily block, its actual cost must be reconciled before a
tail block starts. A tail block is allowed only when its hard run cap fits in
the exact remaining allowance. This uses the daily budget aggressively without
letting price drift or delayed accounting cross `$5.00`.

## Four-Day Allocation

| London date | Primary purchase | Expected / maximum | Decision rule |
|---|---|---:|---|
| Aug 6 | Fresh Qwen source-32 | `$4.2604` actual / `$5.00` | Complete; no more paid calls today |
| Aug 7 | Sealed Qwen history-blind control, then both 128-case budget-model gates | about `$3.30`; hard component caps total `$4.45` | Control first; gates only after its exact mechanics pass; reconcile between components |
| Aug 8 | Diversity-bonus confirmation Block A, then a budget-model mechanics tail if actual headroom remains | about `$4.26`; hard primary cap `$5.00` | Block A owns priority; tail starts only after actual spend is known |
| Aug 9 | Mandatory confirmation Block B, then the same bounded tail rule | about `$4.26`; hard primary cap `$5.00` | Run B after clean A mechanics regardless of A science; combined 64 is sole decision |
| Aug 10 onward | Best mechanics-passing budget model on scaled LLM-native BED work | target `$4.75--$5.00` / hard `$5.00` | Fill with additional preregistered seed blocks, never post-hoc endpoint reruns |

August 6 is closed at `$4.260437`: its leftover was not assigned before the
sealed source run and will not be spent retrospectively merely to hit a number.
From August 7 onward, each day's unallocated tail is assigned before calls to
disjoint reliability or experiment seeds. A posted top-up extends the schedule
but does not raise any daily cap.

## Model Roles

### Qwen 3.7 Plus

Qwen remains the paper-critical generator. It has passed exact structured
serving and produced the existing 96-tree and fresh 32-tree results. It buys
the Aug 7 control and Aug 8-9 staged monotonic-depth test.

### Live Price And Capability Screen

The authenticated OpenRouter model endpoint on August 6 quoted:

| Model | Prompt / completion per 1M | Context | Direct task evidence |
|---|---:|---:|---|
| GPT-5.6 Luna | `$0.10 / $0.60` live API quote | `1,050,000` | exact-10 passed; scale run failed after 1,200 responses |
| DeepSeek V4 Flash 0731 | `$0.09 / $0.18` | `1,048,576` | exact-10 transport passed; one conditioned support was `0/24` valid |

OpenRouter's public Luna page advertises a higher promotional effective rate
than the authenticated models endpoint. Planning therefore uses a fresh live
quote for projection and the provider-reported actual charge for enforcement;
no run depends on the discount persisting.

Artificial Analysis reports Luna at `27` without reasoning, `46` at high
reasoning, and `51` at max reasoning. That distinction matters here: semantic
support generation starts nonreasoning, while thinking remains a naive-baseline
role. Broad benchmark strength cannot erase a conditioned-support failure.

### GPT-5.6 Luna

Luna is a budget candidate, not yet a replacement. Its current price is
attractive and its exact-10 support generation passed, but the scale attempt
had `15/1,200` forced-length responses and a fatal JSON failure. The 128-item
gate tests whether that was a manageable tail or a real reliability limit.

### DeepSeek V4 Flash 0731

DeepSeek-0731 is the cheapest serious candidate, but its exact-10 conditioned
test included a `0/24` valid response. It must pass the same strict
history-conditioned support gate before receiving any policy-scale budget.
Low price cannot compensate for belief-support collapse. If the 128-case gate
passes, however, 0731 becomes the first scale candidate because its output
rate is one third of Luna's live quote. Selection is still by minimum and mean
conditioned-valid support first, then forced exits/parse retries, then cost.

## Concurrency

- use aggregate concurrency `64` for the two 128-item budget-model gates;
- retain the validated Qwen concurrency for sealed paper runs;
- raise toward `128` or `256` only after a mechanics-clean block shows that
  provider throttling and forced exits are not increasing.

The gate workload is too small for concurrency `256` to improve useful
throughput, and changing serving pressure during a sealed replication adds no
scientific value.

## Budget-Model Branch

The two 128-case gates now run immediately after the clean August 7 control,
not on August 10. Each has a `$0.10` hard cap and writes its locally measured
cost back to the shared daily ledger. If one passes, subsequent tail blocks use
that model. If both pass, choose by conditioned-support minimum, conditioned
mean, failure/retry tail, then expected cost. If both fail, close them for
paper work; do not relax parsers or support floors to manufacture a cheap pass.

The first scale purchase is a disjoint-seed conditioned-support stress block,
because the observed failure modes are long-tail failures. Only after that
passes does the model receive a paired policy-efficacy block. This makes cost
per usable belief update, rather than cost per token, the operative Pareto
metric.

## Aug 10 Onward

- If the combined Aug 8-9 result passes, buy an independent replication only
  after the budget-model gate.
- If the combined result is null, do not repeat it unchanged; use the best mechanics-passing
  budget model to investigate a new LLM-native belief-dynamics intervention.
- Keep one experiment decision per day and bank its result before opening the
  next paid block.
