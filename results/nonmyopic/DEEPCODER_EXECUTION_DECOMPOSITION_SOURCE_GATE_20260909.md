# Execution decomposition: source gate and bounded next decision

## Primary-source findings

[ExeDec, ICLR 2024, section 5.2](https://arxiv.org/html/2307.13883v2)
has a few-shot LLM experiment in addition to trained Transformer experiments.
Its LLM variant proposes an execution subgoal and one code step, then replaces
the proposed state with actual execution results before continuing. The ablation
uses stepwise code and execution without subgoal prediction. LLM experiments use
four demonstrations and programs of at most three steps; Pythonic representation
helps. Thus no new model training is intrinsically required to test this idea,
but those results do not validate our four-step structured-output interface or
Bayesian forecasts. They do not establish sequential BED or a depth advantage.

[The released repository](https://github.com/google-deepmind/exedec)
provides a replaceable query_llm hook but warns that its LLM evaluator executes
generated Python. We will not import that evaluator into our host execution path:
only source-typed choices compiled into the pinned interpreter are permitted.

[Shedding Light on Task Decomposition, 2025, section 4.2](https://arxiv.org/html/2503.08738v1)
compares ExeDec with repeated execution-guided synthesis without subgoal guidance.
It finds substantial contributions from repeated synthesis, while decomposition
helps some generalization settings. This is a reason to include an execution-only
control, not evidence that Luna will improve.

## Decision from project evidence

Whole-program execution feedback is closed after its prospective null. That test
revised complete programs, not prefixes. A new stepwise interface changes what the
model conditions on during construction and can coordinate type changes that our
single-edit support expansion cannot express. It is a substantive candidate, not
a reclassification of the failed feedback experiment.

However, the complete length-two control outperforms Luna on the opened panel.
Improving observed-example fitting alone would not justify planning. Three distinct
requirements remain: out-of-sample behavioral coverage, an accurate joint simulator
of answer and future predictions, and a genuine sequential experimental opportunity.
None follows from more synthesis steps. Synthesis length is not BED horizon.

## Implemented zero-call mechanism

`execution_steps.py` validates source choice paths of length zero through four,
enumerates the exact type-valid next syntax menu, and replays all intermediate
values using only public observed inputs. Menus never inspect labels or endpoints.
Full programs require two to four steps. ERROR observations remain in the task;
an intermediate result not matching the final output does not prune the prefix.
No model-estimated intermediate value can replace actual execution.

This is a transport-free mechanism, not an implemented LLM policy. It supplies no
posterior weights, beam selection, probability calibration, or inference claim.
Six focused tests (including existing short-support controls) pass in 1.81 seconds;
scoped lint passes. Tests cover sampled source roundtrips, a coordinated Last/Drop
type change, ERROR retention, history-schema privacy, and invalid choice paths.

## Next dependency, before paid work

Build a paired one-shot runner with four-step construction, a strict next-choice
schema, and source-only demonstrations disjoint from test seeds. Compare subgoal
plus execution with execution-only under the same maximum request/token allocation;
record actual use and do not issue filler requests after completion. Include a
whole-program matched-allocation comparator and the complete two-statement control.
Use the unchanged source law, keep short and ERROR cases, and report results by
source length rather than selecting only cases where the symbolic control fails.

Do not require predicted subgoals to equal execution as a hard rejection rule:
that would be a different algorithm from the cited LLM variant. Record discrepancy
as a diagnostic; only actual interpreter states go into continuation prompts.
No generated Python, true intermediate traces, or held-out outputs enter prompts.

Before dispatch, separately freeze fresh seeds, caps, support construction,
terminal predictive metrics, and advancement criteria. Validate all seals, early
stops, transport failures, no cross-arm context, and budget races. Do not claim this
document alone authorizes a paid runner or a depth sweep. If a new proposal method
passes predictive checks, it must still pass fresh joint transition fidelity and
structural horizon checks before non-myopic evaluation.

## Budget and disposition

Authenticated account credits/usage/balance: 245 / 220.663020549 / 24.336979451.
London Sept9 ledger conservative spend 0.24474207, remaining 4.75525793, including
the unchanged 0.04 uncertain request reservation. No paid calls or new task outcomes
opened in this pass. Previous summary-only turn was no progress; this turn adds a
primary-source decision and tested mechanics. Full research goal remains incomplete.
