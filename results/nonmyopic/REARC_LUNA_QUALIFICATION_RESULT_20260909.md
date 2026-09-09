# Luna-medium RE-ARC qualification: initial execution failure

Frozen implementation: 9bcbab5f. The one-shot run stopped on the first task's
initial proposal gate, before any refresh, symbolic comparison, or target-label
release. The four-task public input collection succeeded without replacement.

One OpenAI-hosted openai/gpt-5.6-luna call, medium reasoning, completed normally:
10,715 prompt tokens, 3,269 completion tokens including 2,659 reasoning tokens.
Cost $0.0066014; no uncertain exposure. This was not a thinking-limit exit.

All four graphs passed the strict format/reference checks and all four executions
returned failure (exit 1). No program matched the first demonstration. Source and
proposal inspection shows each begins with first(I), extracting a row rather
than a grid. Later operations expect grids (fgpartition or compress); the first
graph also selects an object-valued output. This is an interface/executable
proposal failure, not evidence that non-myopic planning fails. The original
execution records retain exit status only, so a specific exception was not banked.

The recorded prefix replays exactly without new calls or executions. Artifact and
implementation hashes verify; forecasts, target labels and downstream analyses
are absent. All 42 RE-ARC tests passed before launch, including synthetic full
and failed-initial replay. No containers remain. The frozen gate stays closed;
no automatic model retry, seed replacement or depth sweep is authorized.

This suggests that a future, separately designed interface should make single-grid
input semantics and executable types harder to misunderstand, with public-example
execution feedback if prospectively permitted. It does not justify claiming that
more reasoning alone fixes spatial induction or that this one task characterizes
the model on the entire benchmark.

Authenticated closing credits/usage/balance: 245/221.313133339/23.686866661.
Conservative London-day spend $0.89485486; remaining $4.10514514.
