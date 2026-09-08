# Luna medium: promising partial observed-fit result

Frozen before responses at 5a03be03. Model openai/gpt-5.6-luna,
standard OpenAI provider, medium reasoning, 16384 total completion tokens.
Same banked cases/seeds/prompts/schema; unsupported temperature omitted.

| Case | Aware compatible / unique | Blind compatible / unique |
|---|---:|---:|
| 0 | 3 / 4 | 0 / 8 |
| 1 | 8 / 8 | 0 / 8 |
| 2 | 1 / 1 | 1 / 8 |
| 3 | 4 / 4 | 0 / 4 |
| Completed-case coverage | 4 / 4 | 1 / 4 |

All eight completed responses cleanly stopped, passed the source grammar, and
used positive reasoning tokens (388 to 3751). Aware total16/17 unique programs
fit both observations; blind1/28. No within-response duplicate programs.
This is a material descriptive improvement over banked DeepSeek nonreasoning
(0 compatible aware cases, including these four), and high reasoning which
returned no programs. Model, reasoning effort and sampling support differ;
this does not isolate the causal effect of reasoning alone.

The ninth attempt, case4 aware, raised IncompleteRead. The runner stopped
without retry as frozen. Case4 is unmeasured, not a failed program. No later
cases or target outcomes opened. The planned eight-pair comparison is incomplete;
this is not a predictive-quality gate pass, calibrated posterior, or BED efficacy.

Independent replay verified each saved request against the prescribed parent
transformation, every response identity/finish/reasoning count, source decoding,
and observed-input execution. Accepted costs sum exactly. Eight response files,
nine attempts; no outcomes file. Terminal SHA
ea89e2c19378f221e2459347ebd78ebe59c975f20490cf63c8d6a4f10bcb35c8.
Prelaunch tests10/10 passed in0.99s; scoped lint passed.

Known completed-response cost $0.01792876. Interrupted attempt retains its
$0.04 uncertainty reservation. Authenticated credits/usage/balance after run:
$245 / $220.410962589 / $24.589037411. Posted London-day spend $0.034268595;
conservative recorded spend including uncertainty $0.069883395, leaving
$4.930116605 within the daily cap. Additional posted usage cannot by itself
identify the interrupted request charge, so its reservation is not released.

Process exited; no retry, fallback, depth sweep or automation restart. Next
scientific question is held-out predictive usefulness of the compatible pools,
not greater planning depth yet. Such follow-up requires a separately defined
comparison; this report leaves the original sealed targets unopened.
