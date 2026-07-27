# DiscoverPhysics Dark-Matter Executable-Support Serving V2 Result

## Decision

**The final one-call serving gate passed every frozen condition.**

This authorizes only a separately implemented and preregistered
simulator-grounded branch policy. The serving response is discarded and
cannot be reused as policy evidence.

## Results

- Requests/HTTP attempts: `1/1`
- Cost: `$0.00846`
- Prompt/completion tokens: `354/505`
- Retries, reasoning tokens, forced exits: `0/0/0`
- Region counts: NE `3`, NW `2`, SW `2`, SE `1`
- Normalized region masses: NE `.42`, NW `.30`, SW `.20`, SE `.08`
- Region-mass L1 error: `.04`
- Geometry types: all four
- Minimum compiled-map RMS: `1.91931`
- Maximum absolute source coordinate: `6.98347`
- Simulator calls: `0`

The integer-weight amendment removed V1's sole arithmetic failure while
preserving the same physical grammar and diversity gates.

## Consequence

The next protocol may:

1. discard this response and generate one fresh executable support;
2. compute branch likelihoods from official simulator trajectories;
3. make eight branch-conditioned executable-support refresh calls;
4. let the model choose branch-specific continuation actions;
5. select myopic and depth-two roots from simulator-grounded values; and
6. evaluate on fresh hidden maps with paired trajectory MSE.

The hidden-map seeds, Monte Carlo noise, policy controls, endpoint, and pass
thresholds must be frozen before the nine responses.

Artifacts:

- Preregistration:
  `results/nonmyopic/DISCOVERPHYSICS_DARK_MATTER_EXECUTABLE_SUPPORT_V2_PREREGISTRATION.md`
- Public serving artifact:
  `results/nonmyopic/discoverphysics_dark_matter_executable_serving_v2/discoverphysics-dark-matter-executable-serving-v2-20260727T013000Z/SERVING.json`
- Public SHA-256:
  `c1ed6546c33606cf12a4ce2b6bceffb330c15910b289f8e6b3645274f4921148`
- Private raw-response SHA-256:
  `3e6e13ed9bcd006b7c99bb4019f4ec8bad0f989be148761cf0a6f536bdd75069`
- OatML use: none
