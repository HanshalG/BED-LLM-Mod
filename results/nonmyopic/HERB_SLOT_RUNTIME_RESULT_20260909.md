# Fixed-slot search and forecast mechanics

Previous goal turn was progress: proposal validation passed but actual search
revealed a separate converter boundary defect. This turn repaired that boundary
in a new runtime; closed study implementations remain unchanged.

`herb_slot_runtime` stores request, exact worker stdout/stderr and process status
before decoding. It verifies manifest/no-key evidence, complete prefix length,
finite nonpositive ordered scores, and the original expansion cap. Individual
conversion failures retain their original index, expression, score and error;
they neither abort a complete search nor trigger refill. Incomplete worker output,
bad bindings, nonfinite scores and exceeded caps still fail closed.

Actual isolated synthetic check, banked in `herb_slot_runtime_smoke_20260909`:
56 slots, 6 conversion failures, 6,872 assignments within the 50,000 cap. The
guidance inputs were null, an ill-arity lbind, and an arity-valid higher-order
graph; they were not benchmark responses. Rejected search expressions included
`__bed_call1(THREE, asobject)` and `__bed_call1(I, recolor)`. These identify the
full-grammar/non-callable terminal mismatch directly in this saved run.
Exact replay reconstructs all 56 slots with no process or API dispatch.

`rearc_slot_forecast` conditions the convertible unique programs on nonempty
observations, then maps the posterior back onto the original attempted slots.
Non-program slots are zero mass after observations, duplicate valid programs
remain canonically deduplicated, and all-invalid pools give explicit unit failure
forecasts. Importantly, execution failure on a future input retains the weight of
a valid program that matched observations. This is the existing finite-pool
posterior convention, not a claim to cover missing true mechanisms.

21 focused tests passed in 1.37s, including actual bank replay with subprocess
dispatch forbidden, tamper rejection, invalid/all-invalid slot forecasts, future
failure mass, all-export arity coverage, and old expression tests. The original
six-call paid failure still replays exactly. No containers remain. API cost $0;
authenticated account usage remains $221.418276889, balance $23.581723111,
conservative London Sep9 spend $0.99999841.

These are engineering checks, not a predictive-transfer pass or non-myopic result.
Next is integrating the new runtime and fixed-slot forecast into a prospectively
frozen qualification controller with source-only adversarial end-to-end tests,
then a fresh disjoint Luna-medium cohort. Do not patch or reopen the closed bank.
The qualification must retain paired aware/blind and symbolic controls, seal
forecasts before targets, and pass predictive transfer before any depth sweep.
The full research objective remains unachieved.
