# Bongard Luna Pre-Outcome Claim Decision Plan

Frozen: 2026-08-06, before any Bongard model response or scientific endpoint.

Strengthened by the August 7 path-dependent-support amendment before any
response. The strongest tier now also requires superiority over fixed-support
depth-two planning.

The 32-task development result contains two distinct prospective questions:

1. Does dynamic depth-two planning improve endpoint Brier over the frozen
   myopic-width policy and its existing controls?
2. Does answer-conditioned VLM regeneration improve over the matched
   same-seed, same-batch history-blind generation control?

These questions must not be substituted for one another after the endpoint is
opened. `scripts/bongard_openworld_luna_claim_report.py` independently replays
the four combined blocks and assigns exactly one claim tier.

## Frozen Tiers

### Full path-dependent LLM-native development signal

Every shared-validity, policy-family, matched-mechanism, and path-dependent
support gate passes. This
permits both prospective development claims and authorizes only a separately
preregistered confirmation. It does not authorize confirmation execution or a
held-out, sealed-test, cross-model, or universal claim.

### Policy signal without matched mechanism

Every shared-validity and policy-family gate passes, but at least one matched
history-blind gate fails. The result may support the prospective dynamic versus
myopic policy comparison. It may not attribute that difference causally to
answer-conditioned belief regeneration and authorizes no confirmation.

### Matched mechanism without policy signal

Every shared-validity and matched history-blind gate passes, but at least one
policy-family gate fails. The result may support a prospective belief-quality
mechanism claim. It is not a non-myopic policy win and authorizes no
confirmation.

### Development null

Neither complete family passes. Report every frozen metric and failed gate,
but make no positive policy or matched-mechanism claim and authorize no
confirmation.

## Gate Families

Shared validity requires exact independent replay of four endpoint-blind
blocks and 32 disjoint tasks, root candidate Brier below the constant-half
baseline, finite endpoint metrics, and unopened confirmation/test data.

The policy family additionally requires the frozen myopic history-change and
tie-margin counts, changes in every block, positive dynamic ranking fidelity
that is no worse than myopic, at least 3% Brier improvement with bootstrap
probability at least 0.80, non-worse log loss, and non-worse Brier than shuffled
continuation.

The matched mechanism family additionally requires at least 12 changed final
histories and one in every block, at least 3% Brier improvement with bootstrap
probability at least 0.80, non-worse log loss, and non-worse ranking fidelity
than history-blind depth two.

The path-dependent-support family requires at least 12 changed final histories
and margin-clearing first actions versus fixed-support depth two, changes in
every block, at least 3% Brier improvement with bootstrap probability at least
0.80, non-worse log loss, and non-worse ranking fidelity. Policy plus matched
mechanism without this family is explicitly reportable but cannot authorize
confirmation or support fixed-support superiority.

The generator rejects missing, extra, non-Boolean, or internally inconsistent
gates; a mismatched independent replay; non-finite report metrics; or any
attempt to promote a partial family to confirmation authorization.
