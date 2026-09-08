# Fresh transition screen: observation-conditioned regeneration recovers one case

Frozen894bd664; four new source cases, two initial examples and a fixed third
query,32 sealed prediction targets per case. Twelve Luna-medium calls completed.
All five-arm forecasts validated and saved before target answers opened.

| Arm | Mean half-Brier | Zero-mass targets /128 |
|---|---:|---:|
| Before third observation (descriptive) | .081743731 | 12 |
| Filter existing support | .250000001 | 32 |
| Insert-stage repair | .164712575 | 21 |
| Regenerate using third observation | .028419010 | 0 |
| Repeat call using old history | .250000001 | 32 |

Regeneration and repeat use equal call counts, output caps, paired seeds and
the same final three-observation filtering. Actual tokens and numerical work
are not matched; the case3 regenerated call uses2236 reasoning tokens versus
672 for the repeat. This is call/cap matched, not equal realized compute.
Before-observation forecasts use fewer observations and are not a fair policy
baseline. Filter/repeat failure includes a prespecified abstention penalty.

All regeneration-vs-repeat gain comes from case3; other three cases tie. On
case3 filter/repeat abstain (Brier1); insertion.65625; regeneration.113676036.
Insertion also mildly worsens case1 (.002600301 versus approximately zero).
Four cases do not establish a population effect, calibrated posterior or a
non-myopic planning result. Regeneration has finite NLL on all128 outputs,
but this does not prove reliable full-support inference beyond this screen.

## Mechanism

Case3 initial proposals compose positive and negative filters, yielding empty
lists; the repeated old-history call returns the same family plus incompatible
parity-filter compositions. The fixed third input is
[[2,-4,7,-8,8],[10,-8,10,0]], observed output[7]. It eliminates the old expanded
pool. Regeneration instead proposes four-stage Take/positive/odd-filter families;
their local expansion produces65 compatible candidates and useful predictions.
This is evidence of observation-conditioned model discovery rather than merely
benefiting from one extra call. It remains a one-case mechanism observation.

Critical planning implication: the revealing output has zero likelihood under
the old support. A lookahead simulator sampling solely that support cannot
anticipate this discovery, regardless of depth. A credible next architecture
needs a validated open-support predictive component or discovery-transition
model; simply increasing rollout count or adding regeneration after execution
does not establish anticipatory model-aware planning. No artificial guarantee
of future truth coverage should enter the simulator.

## Verification and cost

Replay checked all frozen implementation hashes, all12 request seeds and exact
two-versus-three observation routing, source observations, initial saved forecasts,
inserted support, both merged posterior forecasts, target truth and sealed scores.
Forecast SHAd0f4c475421f5d232761fbe6e73cd018ac88ed336327abe739a81cef6125aa1e;
result SHAa3aa0e8cb9cc92012b4419817c0108befdd9983ed1494e2b8e3701221cc53942.
Prelaunch5tests passed2.79s; lint passed. Observation handle was lost during a
turn interruption; PID14973 was independently verified alive and followed to
exit, without restart or duplicate call. Prior turn progress; current turn
completed and verified that experiment.

Cost$0.02538508, no new uncertainty. Live credits/usage/balance:
245/220.458278479/24.541721521. Posted London-day spend$.081584485;
conservative recorded$.117199285 includes prior$.04 uncertainty; allowance
$4.882800715. Goal remains incomplete; automation paused, no depth claim.
