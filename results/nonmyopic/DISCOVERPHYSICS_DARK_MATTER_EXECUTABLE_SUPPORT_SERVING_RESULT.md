# DiscoverPhysics Dark-Matter Executable-Support Serving V1 Result

## Decision

**The exact one-call serving gate failed closed. No simulator-grounded policy
is authorized from V1.**

GPT-5.4 returned eight complete structured hypotheses, but their probability
fields summed to `1.10` rather than one. The strict parser rejected the
response before any simulator or policy endpoint.

## Serving

- Requests/HTTP attempts: `1/1`
- Cost: `$0.009595`
- Prompt/completion tokens: `322/586`
- Retries: `0`
- Reasoning tokens: `0`
- Forced exits: `0`
- Simulator calls: `0`

## Diagnostic Only

After the failure was fixed, a zero-call diagnostic normalized the eight
weights solely to determine whether another defect existed. This does not
rescue or reuse V1.

Every other parser and mechanics check passed:

- region counts: NE `3`, NW `2`, SW `1`, SE `2`;
- all four geometry types present;
- normalized region-mass L1 error: `.16364`;
- minimum compiled-map RMS distance: `1.58659`; and
- maximum absolute source coordinate: `6.74876`.

The sole observed problem is fragile probability arithmetic in the output
grammar.

## Consequence

V1 closes without normalization, repair, or reissue. One final fresh
transport V2 is justified before any endpoint: replace `probability` with a
positive integer `weight` and let exact code normalize. The model, apparatus,
eight-hypothesis count, compiler, physical ranges, diversity gates, cost cap,
and absence of simulator calls remain unchanged. The V1 response must be
discarded.

Artifacts:

- Preregistration:
  `results/nonmyopic/DISCOVERPHYSICS_DARK_MATTER_EXECUTABLE_SUPPORT_SERVING_PREREGISTRATION.md`
- Public failure:
  `results/nonmyopic/discoverphysics_dark_matter_executable_serving/discoverphysics-dark-matter-executable-serving-20260727T010000Z/FAILURE.json`
- Public failure SHA-256:
  `800b373286095cd781b18677dc859158a69a884c4c414994a0a6c5b4c1d91ae6`
- Private raw-response SHA-256:
  `1d45f78052912dc39e6dfe47a976b844f13c05f5ae3c756370e1b988e155d271`
- OatML use: none
