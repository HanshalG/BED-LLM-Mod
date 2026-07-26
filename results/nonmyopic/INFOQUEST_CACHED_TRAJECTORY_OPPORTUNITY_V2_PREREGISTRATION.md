# InfoQuest Cached-Trajectory Opportunity V2 Preregistration

Frozen after V1 failed on one empty released policy utterance and before reading
any record in the fresh V2 split.

## Distinction From V1

V1 is closed and its 80 opportunity IDs are not reused. V2 draws only from the
387 previously untouched IDs in the original effective holdout: the original
388 holdout records minus quarantined ID `4`.

V2 also defines the semantics of rare empty released utterances prospectively.
An empty later policy or simulator utterance is retained as a literal
zero-information action/observation. It contributes no lexical tokens and its
trajectory remains in every denominator. It is never dropped, filled,
reissued, or imputed.

The system message and first policy message must still be nonempty. Every
message content must still be a string. All V1 source, ID, role alternation,
evaluation, reward, substantive metric, and causal-claim restrictions remain
unchanged.

## Fresh Content-Blind Split

1. Start from the sorted original holdout IDs.
2. remove quarantined ID `4`;
3. shuffle with `random.Random(24417)`;
4. assign the first 80 to V2 opportunity;
5. assign the next 30 to V2 development;
6. retain the remaining 277 as V2 holdout.

| Split | Records | Ordered SHA-256 |
| --- | ---: | --- |
| V2 opportunity | 80 | `15a85bd67dc68862b4c77a1ac2ec40f3b921a26ce2362cf16190c7966291c9c1` |
| V2 development | 30 | `7ec12e636bf8e083638a15b9b1ec3ea05087e99e54fd0107e178a28baa1e1642` |
| V2 holdout | 277 | `f99de5f2f33c9499517ac31057e19962817b9ffed217e17d17461eda7028083b` |

Combined V2 split SHA-256:
`acc65f8de579b3f4c5103aed997574769d6338ad367ee60203ad834951d8f4a6`.

The V2 opportunity is disjoint from V1 opportunity, original development,
mechanics, and quarantined ID `4`. V2 development and holdout remain unread.

## Missingness Gates

Across all 480 V2 opportunity trajectories:

1. empty later messages must be at most `.005` of all later policy plus
   simulator messages;
2. trajectories containing any empty later message must be at most `.02`.

Both are conjunctive and precede interpretation of substantive metrics.

## Unchanged Substantive Gates

| Metric | Threshold |
| --- | ---: |
| Multi-turn fraction | at least `.95` |
| At least two initially unresolved | at least `.80` |
| Delayed gain at least two | at least `.75` |
| Mean delayed gain | at least `2.0` |
| Immediate-minus-shifted novel uptake per transition | at least `.25` |
| Trajectories with positive uptake advantage | at least `.60` |
| Task/world cells with turn-count range at least two | at least `.40` |

Tokenization, shifted-answer control, exact three-run grouping, output
redaction, and interpretation are identical to V1. The audit emits IDs,
numeric metrics, reward traces, and hashes only. It cannot establish causal
policy efficacy.

All source, structural, missingness, and substantive gates are conjunctive. A
pass authorizes only a separately preregistered paid ranking-fidelity mechanics
gate on disclosed IDs. A failure closes cached-trajectory InfoQuest
qualification with no further split, parser, missingness, or threshold repair.

OpenRouter calls/cost: `0 / $0`. OatML jobs: `0`.
