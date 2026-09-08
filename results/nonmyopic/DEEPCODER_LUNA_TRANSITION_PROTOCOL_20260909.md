# Fresh observation-transition screen

Four new source cases23100000+i; input seeds24100000+100*i+j, j0..34.
Initial history first2, scheduled observation third, held-out targets final32.
No difficulty rejection or outcome-based query selection. This tests inference
transition quality, not query selection or non-myopic efficacy.

Each case: one initial Luna medium16k proposal call, then two calls with same
paired seed25100000+10*i+1: regeneration sees the full three-observation history;
repeat control sees only the original two. Initial seed25100000+10*i.
Regeneration/repeat order alternates across cases. StandardOpenAI route,
same source schema/prompt, no temperature. At most12calls, cap$.60, perattempt$.04.
Carry prior uncertain$.04 reservation without release; daily account-wide$5.
No retries/fallback, final live reauthorization immediately before each attempt.

Initial raw candidates receive one type-preserving local expansion. Save their
restricted-prior prediction BEFORE opening the third answer. Precompute inserted
stage alternatives using only the initial history. After the scheduled answer,
compare filter-only, insertion-repair, regenerated union with old support, and
repeat-control union with old support. New raw pools receive the same one-edit
expansion conditioned on the history their call received; final weighting for
all four transition arms uses the same full three-observation history. Regeneration
and repeat have equal call/token caps, though actual token and numerical work
may differ. Before-observation baseline has fewer observations and is descriptive.

Every target forecast for all cases/arms must be saved and validated before any
target answer is loaded. Primary descriptive contrast is regenerated vs repeat
control mean half-Brier; also filter/repair, zero-mass counts, abstentions and NLL.
Abstention loss1, all cases retained. Four cases authorize no significance,
calibration or depth claim and have no post-hoc pass threshold. Output screen
does not validate simulated-answer transitions or model-aware non-myopic planning.
Freeze code/protocol before the new responses. Preserve all prior artifacts.
