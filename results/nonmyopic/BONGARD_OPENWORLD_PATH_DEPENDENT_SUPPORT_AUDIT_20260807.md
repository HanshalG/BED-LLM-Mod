# Bongard Path-Dependent Support Audit

Date: 2026-08-07. Model calls: 0. Cost: $0. Scientific endpoints opened: 0.

## Finding

The unopened Bongard design has exact call parity and fixed response width:
every policy shares one root belief, all answer-conditioned branches, and all
deduplicated terminal beliefs; every belief contains exactly ten hypotheses.
The matched history-blind control receives one same-seed adjacent draw per
dynamic branch and analytically conditions it on the same simulated answer.
Root outcome probabilities weight both planners identically. These checks rule
out extra calls, support count, and branch-probability weighting as explanations
for a dynamic result.

One claim gap remained. The strongest development tier required dynamic depth
two to beat myopic width and matched history-blind regeneration, but required
only non-inferiority to fixed-support depth two. It could therefore authorize
confirmation when dynamic and fixed chose identical paths or had equal endpoint
quality. That would not establish that anticipating the LLM's path-dependent
support regeneration adds value beyond conventional fixed-support lookahead.

## Pre-Response Correction

The amendment
`BONGARD_OPENWORLD_LUNA_PATH_DEPENDENT_CLAIM_AMENDMENT.md`, SHA-256
`3dc22154312b93465e2b7d308a76f9de800d145d219189b77f3a0262f83f32a4`,
adds a separate path-dependent-support family.

Development now requires at least 12 dynamic/fixed changed histories, 12
margin-clearing first-action changes, changes in every block, at least 3%
relative Brier improvement, paired bootstrap improvement probability at least
0.80, non-worse log loss, and non-worse ranking fidelity. Confirmation requires
the corresponding 24-change thresholds and a paired tree-bootstrap 95% upper
bound below zero.

The new strongest tier is
`full_path_dependent_llm_native_development_signal`. Policy and matched
regeneration can both pass while fixed-support superiority fails; that result
is banked as
`policy_and_matched_regeneration_without_fixed_support_superiority` and cannot
authorize confirmation.

No task, image, prompt, response, policy, call, seed, date, budget, endpoint, or
existing myopic/history-blind threshold changed.

## Frozen Evidence

- Development interface: `bongard-openworld-luna-vlm-development32-7`.
- Development manifest SHA-256:
  `a649a76926b84cebc2a6e4f5b782d451dddb8634207836cbb9901543415d9ce1`.
- Claim-report interface: `bongard-openworld-luna-claim-report-2`.
- Confirmation freeze interface: `bongard-openworld-luna-confirmation64-freeze-3`.
- Confirmation V3 manifest SHA-256:
  `8a6dd0879ee63b38a016ca954285563ec1bd1964d018ae6af79e24547aa2d284`.
- Confirmation execution core SHA-256:
  `00995532e8a8e1e7bd4c27e4513c0332e5a9aa16834c79fc7973ee689a0a4a33`.
- Development claim finalizer SHA-256:
  `89fd4f64587908fbeaaffe6aa0c4db28824ca83cc10e5ca020d03ef89d2a24f2`.
- Full Bongard suite: 114 passed in 44.37 seconds.
- Independent confirmation protocol: all 12 checks pass.
- Independent execution bindings: pass.
- Authenticated August 10 preflight: `ready_without_paid_calls`, live balance
  `$24.886393846`, component-cap sum `$2.00`, calls 0, files written 0.

The frozen paid path remains August 10 Luna serving plus mechanics. The stronger
claim family affects only later development interpretation and conditional
confirmation authorization.
