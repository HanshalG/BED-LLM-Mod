# Paired repair: executable-output failure before scoring

Frozen executor/protocol commit050ac688. Both Luna medium calls returned normal
stop responses with accepted receipts, total cost **$.01090445**. Control cost
.00594880, refresh .00495565. No retry or new uncertain exposure. Process exited.

Control returned one syntactically valid expression. Refresh returned three;
its second expression has an unmatched closing parenthesis. Strict all-candidate
decoding raised ValueError before the forecast seal and before this runner read
the old target labels. Saved status is failed_closed, calls2, endpoints_opened
false. There is no forecasts.json or outcomes.json in the new run directory.
Banked responses remain intact. No expression was repaired, discarded to rescue
the run, or scored selectively. The old targets were previously opened development
data regardless of this runner's ordering.

This does **not** measure the value of an extra observation, establish a predictive
null, or invalidate numerical scale calibration. It shows that free-form formula
strings remain a failure surface even after strict outer JSON and residual/domain
feedback. Close this exact paired diagnostic. The next interface should compile a
bounded typed expression graph rather than ask the model to balance long formula
strings; freeze that representation before responses, test arity/topology/domain
handling, and retain equal-compute new-observation controls. Do not auto-correct
these responses or claim a non-myopic result. No depth run is authorized.

Prelaunch15tests1.21s/lint; postrun regression replays exact requests, all immutable
bindings, both receipts and the decoding failure without opening target labels.
The broader goal remains unachieved. The preceding final response restated banked
results (no progress); this turn froze/tested a new interface and completed a
measured, diagnostic failure that changes the next action.

Live credits/usage/balance245/221.234591189/23.765408811. London Sept9 conservative
spend.81631271 includes the old.04 uncertainty, remaining4.18368729. No cluster,
automation, protected runtime or closed experiment was changed.
