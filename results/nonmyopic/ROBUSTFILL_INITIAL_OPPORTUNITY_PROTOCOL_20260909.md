# Initial-only RobustFill empirical opportunity diagnostic

Development only: tasks1/10 remain opened. Do not load any labels except the saved
e1 initial pair. Use hash-pinned old forecasts for e2..10 input strings only; old
forecast rows must not enter the new prior. Keep e2..7 queries/e8..10 targets and B3.

Prior fixed before this diagnostic: uniform concatenation length1..3, IID uniform
over the 162043 source atom expressions from the banked coverage audit, including
syntax multiplicities, empty pieces and optional case transforms. This is a new
restricted source prior, not ExeDec's training distribution or all RobustFill.
Compute exact conditional syntax counts from the initial output using prefix dynamic
programming. Draw256 IID conditional programs, seed50100000/50100001, without output
rejection or subsequent-label filtering. Save their nine-column predictions first.

Evaluate ordinary receding h1/h2/h3, random and fixed open-loop on the same finite
empirical law with existing5s/100000node per-plan and120s panel caps. Incomplete is
not a null. Report all outcomes without changing seeds, prior, roles or capacities.
The exact plans apply only to this sampled law, not the full conditional prior.
No real-path scores, API calls or paid authority. A positive internal gap would
require independent-panel robustness and true predictive calibration; a zero gap
would make this configuration a poor next paid depth experiment, not prove a general
impossibility. This diagnostic cannot establish the LLM's necessity.
