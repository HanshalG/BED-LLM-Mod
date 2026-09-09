# Higher reasoning does not qualify the correction role

Frozen98369ed0;all16requests completed cleanly, cost$.04430910. No retries, partial
pool acceptance or new uncertainty. Source/request/forecast/endpoint/receipt replay
valid; forecast675714ea17651df0b795a77c2cdda34e077f02559ccb187310b27dfc3dae2514.

| Predictor | Mean MSE |
|---|---:|
| Calibrated base | .0306212073 |
| Medium old-history | .0263529876 |
| Medium refreshed | .0303486858 |
| High old-history | .0306223484 |
| High refreshed | .0285485018 |
| Numerical ridge | .0046303041 |

Both efforts fail unchanged gates. High's new-history improvement is6.77%, below
10%, and neither effort has any paired gain>.01. High refreshed is5.93% better
than medium refreshed but6.17times ridge loss. Medium refresh is worse than its
old-history control. This4case diagnostic is not a significance claim or evidence
that high reasoning never helps; it rejects qualification of this interface.

| Effort | Calls | Prompt tokens | Completion | Reasoning | Cost |
|---|---:|---:|---:|---:|---:|
| Medium | 8 | 7787 | 9267 | 8960 | .01288575 |
| High | 8 | 7787 | 24715 | 24439 | .03142335 |

High used2.73times reasoning tokens and2.44times cost. Maximum completion was2147
medium and7084high, both well below16384; no token-limit failure explains the null.
All requested payloads differ only in effort within matched history/case pairs.

All generated corrections in both efforts are constants. High's case2 +.84 yields
the modest gain; none represents a secondary-input dependency. Case1 high refresh
even proposes multiplication by0, which compiles but is excluded as nonpositive;
the valid base remains, without a fallback or changed rule. All diagnostics retained.

## Decision

Close this exact comparison. Do not run max effort, more seeds or a planner on the
same numeric correction interface. These tasks strip away the strongest observed
LLM contribution, semantic prior knowledge, and ask it to compete with numerical
regression on a low-dimensional curve. The restricted representation may contribute
to poor outputs, but the current evidence does not uniquely identify model versus
prompt/representation failure. Do not assert a provider bug or global incapability.

Next revisit source selection for a genuinely semantic executable task, such as
specification-grounded code diagnosis/repair, before more paid numeric fitting.
Audit prior banked code tasks to avoid repeating context-free DeepCoder/string
nulls. Require a public specification, coherent executable candidate worlds, a
nonprivileged observation channel and genuine sequential query opportunity. A
source audit is not benchmark adoption or a paid authorization. The full target
remains LLM-native non-myopic BED with productive compute-matched controls, not a
smaller synthetic positive or monotonicity under an oracle.

15prelaunch tests1.43s/lint; banked replay regression added after outcome. All
processes exited. Previous turn was progress (semantic qualification null); current
turn supplies a matched reasoning comparison and changes the next source decision.
No positive non-myopic result is established; goal remains unachieved.

Authenticated account245/221.306531939/23.693468061; conservative LondonSept9spend
.88825346 includes old.04uncertainty, remaining4.11174654. No cluster/automation changes.
