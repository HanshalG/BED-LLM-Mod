# Bongard Path-Dependent Support Claim Amendment

Date frozen: 2026-08-07, before any Bongard model response or scientific
endpoint.

## Identifiability Gap

The development protocol already compares dynamic depth-two planning against
myopic width, fixed-support depth two, shuffled continuation values, matched
history-blind regeneration, and random selection. Its strongest claim tier
requires dynamic to beat myopic and history-blind controls, but requires only
non-inferiority to fixed-support depth two.

That boundary is insufficient for the intended headline. Dynamic and fixed
depth-two planning could choose the same paths, or fixed support could match
dynamic endpoint quality, while the old conjunction still passed. Such a
result would not show that anticipating the LLM's answer-conditioned support
regeneration adds value beyond conventional non-myopic planning on the root
support.

## New Strongest Tier

Preserve every existing task, response, policy, control, seed, model call,
budget, endpoint, and threshold. Add a separate `path_dependent_support` gate
family. On 32 development tasks it requires:

1. at least 12 dynamic final histories differ from fixed-support depth two;
2. at least 12 first-action differences clear the existing `1e-6`-nat margin
   under the dynamic score;
3. dynamic and fixed differ in every eight-task execution block;
4. dynamic improves endpoint Brier over fixed by at least 3% relatively;
5. the paired task bootstrap probability of Brier improvement is at least
   0.80;
6. dynamic endpoint log loss is no worse than fixed; and
7. dynamic score ranking fidelity is no worse than fixed.

The new strongest claim tier is
`full_path_dependent_llm_native_development_signal`. It requires the complete
shared-validity, dynamic-versus-myopic policy, matched history-blind mechanism,
and path-dependent-support families. Only this tier may authorize the already
frozen confirmation task set.

If the policy and matched-mechanism families pass but path-dependent support
does not, report
`policy_and_matched_regeneration_without_fixed_support_superiority`. That tier
may describe both registered development comparisons but cannot authorize
confirmation or claim an advantage over fixed-support planning.

## Confirmation

The 64-task confirmation applies the same family with confirmatory strength:
24 changed histories, 24 margin-clearing first-action changes, differences in
every block, at least 3% relative Brier improvement, paired tree-bootstrap 95%
upper bound below zero, non-worse log loss, and non-worse ranking fidelity.

This amendment strengthens the claim boundary before responses. It does not
use outcome data, relax a gate, or create a rescue path.
