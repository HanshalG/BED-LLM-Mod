# KnowU Dynamic-Support Mechanics V4 Result

Date: 2026-07-26

Status: **all frozen gates passed**

## Result

The exact six-world KnowU first-link gate completed with the cached V3 initial
support and 36 new V4 calls:

- composite accepted requests / HTTP attempts: 42 / 42
- reasoning tokens, retries, and forced exits: 0
- composite cost: $0.357380, below the $0.75 cap
- initial truth missing: 2 of 6 worlds
- worlds where a missing truth entered after an atomic question: 1 of 2
- truth-entry events: 2
- mean within-world support-gain range: 10.0 points

The positive event was `T1W3` in the computer-purchase family. Its initial
truth-support score was 68. Asking about operating system raised it to 78;
asking about shopping platform raised it to 82. Budget and use-case questions
scored 66 and 69. Thus the effect is not merely a binary threshold crossing:
the two successful branches gained 10 and 14 points while the other branches
lost 2 or gained 1.

The other initially missing world, `T1W2`, remained absent in all four
branches. The initial support compressed a dual coding/gaming preference into
one inexpensive gaming laptop. None of its four generated questions asked
whether separate use cases required distinct machines, so refreshed support
kept the same structural error. Its best branch improved only from 55 to 60.

The leave-notice worlds and the other computer world already contained truth
initially, so they do not test support entry.

## Interpretation

This passes the narrow first causal link: a real semantic answer can change
the LLM's generated support so that a previously missing true preference state
appears. It also exposes the planning problem sharply: useful entry depends on
asking the right latent dimension, and current question generation can remain
trapped inside its initial support.

This is not yet evidence that non-myopic selection works. It is one entry
world out of six, the frozen threshold is 70, and GPT-5.4 generated the
supports, simulated the user, and judged truth coverage. The next gate is an
independent-family rescore of the frozen 30 supports, followed only if that
passes by candidate-strategy ranking fidelity. No depth sweep is authorized.

Public result SHA-256:
`bf433f1ff1b6f9ec2acbf359f384922380dfa21b7a06ff4be65c60ec82e76039`.
Private raw SHA-256:
`0f12315cb22ca54cf923c9075b7397dc9cd38a814c7c5762120ec0928ef5cf97`.
