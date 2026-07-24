# PAPRIKA Instrument Support-Recall Result

## Decision

The serving smoke failed its frozen complementary-label consistency gate. No
hypothetical branches, activity cases, ranker scores, or confirmation targets
were evaluated. This exact instrument interface is closed.

## Failure

The run stopped after 16 physical requests and `$0.01683000`, with zero
reasoning tokens, retries, forced exits, parsing failures, or runtime failures.
It had completed:

- two direct and two complementary prefix-label calls;
- six initial support-generation calls;
- two candidate-question calls; and
- two direct and two complementary current-support-label calls.

The prefix labels agreed. The current-support labels did not. After reversing
both axes of the No-framed responses and complementing their booleans, 20 of 96
item-question labels differed from the Yes-framed responses. The code's axis
mapping was verified locally against the frozen raw responses.

Examples include disagreement over:

- whether timpani and tubular bells are tuned to pitches for melody;
- whether triangle, tambourine, and several drums are held while played;
- whether tambourine uses a stretched membrane;
- whether viola is played under the chin or shoulder;
- whether ukulele, sitar, and acoustic bass guitar have fretted necks; and
- whether double bass is commonly bowed rather than plucked.

Some are genuinely usage-dependent, while several are ordinary classification
errors. Either way, they show that free-form semantic binary labels are not a
stable enough response/likelihood layer for this prospective comparison.

Public failure artifact SHA-256:
`a682fb7836556671139a3e8ad14e0feb1a2f0ca5b4e447945db967e6051e155b`.

Private raw checkpoint SHA-256:
`2cbc49e2fc56d54e861321df615c92675f6c468575e859d6d3d4a4ab6f2615d1`.

## Interpretation

The clothing and instrument attempts now agree on the same bottleneck. Open
world support generation is usable, but an unconstrained LLM semantic
environment is too inconsistent to define the realized branch cleanly. A
target-blind support scorer cannot be evaluated honestly if the answer that
selects the branch changes under an equivalent complementary framing.

The next route should therefore keep the LLM in the irreducible hypothesis
generation and/or support-scoring role while obtaining observations from an
external deterministic source: a retrieval corpus, simulator, database, or
released structured world. Another natural-language yes/no environment or
post-hoc consensus repair is not authorized by this result.

## Budget

The project ledger is `$70.34495466` spent with `$35.03984803` headroom. The
live account has `$60.03984804` remaining, or `$35.03984804` above the protected
`$25` Monday reserve. OatML was not used.
