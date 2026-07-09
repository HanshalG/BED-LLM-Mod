# Qualitative Location Strategy Examples

Smoke-only artifact from one paired support-grid pilot trial (`n=1`). Use this to
illustrate the qualitative mechanism and expected report shape; do not treat it as
powered Phase 4 evidence.

## Example 1: trial 0 / StrategyEIG-d5

- RMSE delta vs EIG: `0.0`
- Truth-log-prob delta vs EIG: `0.5804333506993524`
- Plot: `results/location_qualitative/support_grid_smoke_constrained_qualitative_example_1.png`

### Round 1

- Selected EIG: `1.190706582706838`
- Root query: `[2.2, 1.76]`

Verify the secondary cluster to determine if the posterior is multi-modal. If a
significant signal is detected, refine the position of the secondary source. If the
reading is low, begin a step-wise movement towards the primary cluster at [-1.43, 0.0].

### Round 2

- Selected EIG: `0.6622659899539681`
- Root query: `[1.76, 1.55]`

Move towards the primary high-probability source cluster near [-1.43, 0.0] by taking a
step in that direction to intercept the signal gradient.

### Round 3

- Selected EIG: `0.5602472700178349`
- Root query: `[1.3108078195403996, 1.3303949339975287]`

Move towards the most probable source location at approximately [-1.4, 0.0] by taking
steps of 0.5 in that direction. If a signal increase is observed, transition to a local
grid search to refine the source position; otherwise, continue the trajectory.

### Round 4

- Selected EIG: `0.5033820164191052`
- Root query: `[0.8616156390807992, 1.1107898679950574]`

Navigate towards the primary source cluster near [-1.4, 0.0] by moving in large steps in
that direction. Once the signal strength begins to rise, switch to a local refinement
strategy to pinpoint the exact coordinates.

### Round 5

- Selected EIG: `0.6946492448110195`
- Root query: `[0.4117, 0.8927]`

Navigate toward the primary hypothesis cluster at [-1.43, 0.0] by taking steps of 0.5 in
that direction. Once the signal strength significantly exceeds the baseline, switch to a
local search to refine the source location.

### Round 6

- Selected EIG: `0.4169578134319121`
- Root query: `[-0.0382, 0.6746]`

Proceed towards the primary source candidate at approximately [-1.43, 0.0] by moving in
0.5-unit steps along the vector connecting the current position to that location. Once
the signal strength rises significantly above the baseline, transition to a local
refinement strategy using a fine-grained search to pinpoint the exact source
coordinates.

## Example 2: trial 0 / StrategyEIG-d3

- RMSE delta vs EIG: `2.380797689164583`
- Truth-log-prob delta vs EIG: `-0.021705492376639768`
- Plot: `results/location_qualitative/support_grid_smoke_constrained_qualitative_example_2.png`

### Round 1

- Selected EIG: `0.8834628790525025`
- Root query: `[-1.43, 0.0]`

Exploit the high-probability region by attempting to locate the peak of the primary
source. If the initial reading is high, perform local search steps to refine the
estimate. If the reading is low, proceed with a stepping traversal towards the secondary
clusters.

### Round 2

- Selected EIG: `1.7722103837214398e-06`
- Root query: `[-0.93, 0.0]`

Move towards the centroid of the high-probability source clusters to bridge the distance
and detect the signal gradient. Once the signal strength rises significantly above the
baseline, transition to a local gradient-ascent search to pinpoint the source
coordinates.

### Round 3

- Selected EIG: `0.0003865406394558124`
- Root query: `[-0.43, 0.0]`

Perform a directed search by moving towards the centroid of the current high-probability
mass. Once the signal deviates significantly from the baseline, switch to a local
gradient-ascent strategy to isolate the specific source location and resolve the
symmetry in the y-coordinates.

### Round 4

- Selected EIG: `0.11436542582736786`
- Root query: `[0.07, 0.0]`

Navigate towards the high-probability source cluster centered around x=2.2 by moving in
increments towards the estimated coordinates. Once the signal strength rises above the
baseline, implement a local refinement strategy to distinguish between the possible
source locations in the positive and negative y quadrants.

### Round 5

- Selected EIG: `0.042691868221904905`
- Root query: `[0.57, 0.0]`

Incrementally advance along the x-axis toward the cluster of high-probability sources
near x=2.2. Once a signal increase is detected, perform a local search in the
y-dimension to distinguish between the positive and negative y-axis hypotheses.

### Round 6

- Selected EIG: `-2.555041681284731e-06`
- Root query: `[0.57, 0.0]`

Incrementally advance along the x-axis toward the cluster of high-probability sources
near x=2.2. Once a signal increase is detected, perform a local search in the
y-dimension to distinguish between the positive and negative y-axis hypotheses.

## Example 3: trial 0 / StrategyEIG-d1

- RMSE delta vs EIG: `2.380797689164583`
- Truth-log-prob delta vs EIG: `-0.029109955351131145`
- Plot: `results/location_qualitative/support_grid_smoke_constrained_qualitative_example_3.png`

### Round 1

- Selected EIG: `0.5691449936663687`
- Root query: `[-1.43, 0.0]`

Exploit the high-probability region by attempting to locate the peak of the primary
source. If the initial reading is high, perform local search steps to refine the
estimate. If the reading is low, proceed with a stepping traversal towards the secondary
clusters.

### Round 2

- Selected EIG: `-2.1926904736346842e-14`
- Root query: `[-1.43, 0.0]`

Exploit the high-probability region by attempting to locate the peak of the primary
source. If the initial reading is high, perform local search steps to refine the
estimate. If the reading is low, proceed with a stepping traversal towards the secondary
clusters.

### Round 3

- Selected EIG: `6.559641718695275e-12`
- Root query: `[-0.93, 0.0]`

Perform a directed search towards the high-probability regions by stepping incrementally
towards the suspected source coordinates, using the gradient of the signal to home in on
the source centers once the signal exceeds the baseline.

### Round 4

- Selected EIG: `-5.586198170703938e-12`
- Root query: `[-0.93, 0.0]`

Perform a directed search towards the high-probability regions by stepping incrementally
towards the suspected source coordinates, using the gradient of the signal to home in on
the source centers once the signal exceeds the baseline.

### Round 5

- Selected EIG: `5.083162779584427e-10`
- Root query: `[-0.43, 0.0]`

Advance towards the estimated source centroid in 0.5-unit steps to find the signal's
detection threshold. Once the signal rises, perform a local search to separate the two
likely source candidates by exploring the y-axis.

### Round 6

- Selected EIG: `-3.21365345445912e-09`
- Root query: `[-0.43, 0.0]`

Advance towards the estimated source centroid in 0.5-unit steps to find the signal's
detection threshold. Once the signal rises, perform a local search to separate the two
likely source candidates by exploring the y-axis.
