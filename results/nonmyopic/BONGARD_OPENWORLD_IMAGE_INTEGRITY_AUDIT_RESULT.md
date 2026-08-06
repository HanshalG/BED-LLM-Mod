# Bongard-OpenWorld Image Integrity Audit Result

Date: 2026-08-06

## Decision

`image_integrity_pass`; `scientific_opportunity_status=untested`.

The complete official archive matches the bound metadata and all four frozen
mechanics tasks are image-decodable under the opaque sequential interface.
This result banks the first complete official archive hash. It does not
authorize model calls or establish a non-myopic advantage.

## Frozen Evidence

- Archive bytes: 5,124,375,111
- Archive SHA-256:
  `5ab838cffd8c1be6080e232b9fa4d9ca824d213371625abe59d984745f694d25`
- ZIP entries: 15,151 (14,140 files and 1,011 directories)
- Metadata file-set difference: 0 missing, 0 extra
- Duplicate entries: 0
- Zero-size files: 0
- Unsafe member paths: 0
- Full CRC scan: pass
- Uncompressed image bytes scanned: 5,181,779,641
- Mechanics images verified and fully loaded: 56/56
- Mechanics formats: 52 JPEG, 2 PNG, 2 WEBP
- Mechanics dimensions: width 300--2,500; height 250--3,750
- Opaque/truth-free public mechanics payload: pass
- Integrity gates: 11/11 pass
- Model calls: 0
- OpenRouter cost: $0

Public manifest:
`results/nonmyopic/bongard_openworld_image_integrity_audit/bongard-openworld-image-integrity-audit-20260806/MANIFEST.json`

Manifest SHA-256:
`239943ae789ebdc2c0a03577a02b04890c6d00f50ce45639c5defc1624ccee96`

## Qualitative Mechanics Review

Private contact sheets were inspected for the four mechanics tasks. All images
are legible and there are no obvious duplicate panels. The tasks include both
clean category discrimination and genuinely semantic boundary cases. In one,
checkerboard-like mats or temporary surfaces must be distinguished from the
target use as flooring. In another, the rule is a conjunction of medium,
object scale, and subjective attribute rather than an object category alone.

This is useful but risky evidence: it gives the LLM irreducible compositional
work, while also making fixed dataset labels imperfect semantic likelihoods.
The mechanics smoke must therefore measure prediction calibration and
hypothesis coverage, not only exact endpoint accuracy.

## Next Gate

After the already frozen Aug 7--9 OpenRouter sequence, run a validation-only
four-task multimodal mechanics smoke. Use the same image/call cache across
myopic, fixed-support depth two, path-dependent refreshed-support depth two,
and random controls. Require exact structured outputs, executable opaque image
IDs, branch-dependent hypothesis regeneration, and a nonsaturated externally
labelled endpoint before opening development tasks.
