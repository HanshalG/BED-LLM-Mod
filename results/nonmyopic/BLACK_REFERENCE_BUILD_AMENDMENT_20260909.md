# Build-only correction before any formatter/reference execution

V1 failed at its first Docker build: BuildKit resolved FROM sha256:<local image ID>
as a registry image name. No pre-fix inputs were run and no fixed source fetched.
Keep BLACK_REFERENCE_DISAGREEMENT_20260909.json unchanged as the failure record.

V2 uses the existing local parent tag, explicitly verifies its image ID matches
the original frozen017df47b image before each build, and writes a distinct result.
Original protocol,96inputs, source revisions, observation handling, eligibility
criteria and zero paid calls are unchanged. This corrects instrumentation before
outcomes; it is not a threshold adjustment or scientific null retry.
