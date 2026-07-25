# Animals Support-Retention Holdout Preregistration

Date: 2026-07-25

## Development Disclosure

The expected-support-size development gate failed and is not relabeled.
After its 20 aligned endpoints were opened, expected current-support retention
was selected from the preregistered controls. It achieved realized target
coverage `.65` versus `.55` for immediate EIG (`2/18/0`) and recovered four
initially omitted targets versus two. Its within-state pairwise accuracy was
`.9167` versus `.3333`, but only six binary-outcome pairs were informative.

This is method selection on development data, not evidence. On the earlier
matching stratified-prior development artifact, retention improved
model-averaged target coverage by `+.0643` (`7/11/2`); on the new aligned
development it improved that diagnostic by `+.0980` (`6/12/2`). Older
mismatched protocols range from `-.0074` to `+.0098`, so generalization is
uncertain.

## Frozen Holdout

- Targets: all 60 previously sealed names in the frozen seed-24285 target
  pool, SHA-256
  `f1357649054e41150580201b8ad2318e30fb752ae7adb647e85873f7115a6715`.
- Model: `google/gemma-4-26b-a4b-it`, non-thinking.
- Config SHA-256:
  `0f291564e034103be1021dc2fb1a46d47a5be3f4b0719f5781e171d1f6ace052`.
- One target-independent fixed prehistory question per state.
- Three retained roots from five requested candidates.
- Both Yes and No branches use the production regenerate, animal-name
  validation, semantic filtering, old-support carryover, and retry path.
- A stateless target-blind semantic table over all 126 prior animals defines
  both likelihoods and the realized hidden-target answer.
- All selectors share the exact current belief, roots, semantic tables, and
  six regenerated branches.

The primary selector maximizes expected retained probability mass from the
current generated support:

```text
p(Yes) * retained_mass(Yes) + p(No) * retained_mass(No)
```

The primary control is immediate EIG on the same current support. Seeded random
is a required secondary control. Expected support size remains diagnostic only.

The sole serving amendment is an output cap increase from 4,096 to 16,384
tokens. Two of 9,948 non-thinking development requests reached the old cap.
No prompt, model, temperature, candidate width, belief-update rule, score,
endpoint, target order, or seed changes. Any length exit under the new cap
still fails the holdout.

## Endpoints And Gates

The primary endpoint is realized inclusion of the hidden target in the
regenerated support corresponding to its aligned semantic answer. The primary
paired estimate is support retention minus immediate EIG across all 60
targets.

Every gate is conjunctive:

1. all 60 states complete with three candidates;
2. retention and EIG select different roots on at least 20 states;
3. immediate-EIG coverage lies between `.05` and `.95`;
4. mean realized retention-minus-EIG coverage is positive;
5. its paired bootstrap 95% lower bound is strictly positive;
6. the exact one-sided sign-test value is at most `.05`;
7. wins exceed losses;
8. model-averaged expected-coverage gain is positive;
9. retention recovers at least as many initially omitted targets as EIG;
10. retention has positive mean realized gain over seeded random and more wins
    than losses;
11. zero reasoning tokens, forced exits, and retries;
12. total cost is at most `$2.00`.

Uniform realized truth mass, support size, correct within-state pairwise
accuracy, support expansion, and candidate-level rankings are diagnostics.
They cannot substitute for the primary gates.

Passing supports a fresh LLM-native first-link claim and authorizes a separately
preregistered multi-round receding policy. Failure closes this exact
support-retention interface. There is no parser repair, threshold tuning,
target subset, or rerun after endpoint access.

Projected cost is `$1.00`. The authenticated OpenRouter balance and local
ledger must preserve the protected `$25` reserve immediately before launch.
No OatML resources are authorized.
