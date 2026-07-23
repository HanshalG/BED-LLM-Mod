# UCI Thyroid Utility-Grounded GPT-5.4 Mini Failure Result

The frozen mechanism ablation did not produce a paired policy endpoint. Its S0
serving smoke passed, but S1 failed closed at logical cell 62 after the single
registered correction attempt. There is no replacement seed or rerun.

## Frozen treatment

The model received, for every root outcome and legal continuation, the empirical
posterior-predictive expected class entropy and equivalent one-step information
gain. These cards used the deployed history and 7,200-row prior, not the held-out
patient or a realized future outcome. Machine-fixed roots, exact depth-two strategy
scoring, matched-random and exact controls, and all gates were unchanged.

## S0 mechanics

- All 12 late-state cells passed on the first attempt.
- History lengths 0--6 were covered; every continuation was legal and no policy
  repeated its root.
- GPT selected the minimum-entropy card on 98/154 branches and chose `query:tsh`
  after initial blood collection.
- Usage: 12 requests, 92,691 prompt tokens, 1,383 completion tokens, zero reasoning
  tokens or forced exits, and `$0.07176735`.

## S1 failure

S1 accepted 55 logical cells before stopping. Four invalid physical responses were
observed: cell 0 attempt 0, cell 60 attempt 0, and both attempts at cell 62. At the
terminal cell, the history was:

```text
collect blood -> TSH -> pregnant -> goitre -> T4U -> thyroid surgery
```

The `query:age` branches had essentially zero class entropy and many tied legal
continuations. Attempt 0 repeated the root `query:age` for outcomes 2 and 3. The
registered correction changed outcome 2 but again returned `query:age` for outcome
3, which was absent from the shrinking legal menu. The run therefore failed closed
before aggregate traces or any scientific endpoint were written.

The quarantined accepted prefix is mechanism-only:

- utility-card adherence was 493/708 branches (69.6%);
- all 8/8 observed initial decisions selected blood collection; and
- all 8/8 collection policies selected TSH as the continuation.

These counts are not a paired policy result. They show that calibrated utility fixed
the initial continuation/root behavior, while free-form named serialization remained
brittle in low-entropy late states. A deterministic card compiler passed the frozen
pipeline and independent replay in dry run, but that is an exact-control diagnostic,
not LLM-policy evidence.

S1 usage was 59 physical requests, 458,656 prompt tokens, 7,177 completion tokens,
zero reasoning tokens or forced exits, and `$0.21921330`. S0 plus S1 cost
`$0.29098065`; project spend is `$35.21476396 / $110`, leaving `$74.78523604`.

## Conclusion

This test supports the utility-grounding diagnosis but does not establish a repaired
LLM policy. The next architectural step should compile or constrain legal
continuations before exact verification, rather than asking the model to reproduce
large shrinking action-name maps. Such a method must be separately preregistered and
must preserve an ablation that reveals how much policy value still comes from the LLM.
