# InfoQuest Cached-Answer Ranking Diagnostic Result

The frozen post hoc diagnostic completed cleanly and failed its scientific
gates. Removing fresh simulator variance did not rescue semantic-partition
exact EIG.

## Protocol

- run ID: `infoquest-cached-answer-mechanics-20260726T034528Z`;
- exact 6 physical requests and 6 HTTP attempts;
- zero new support, partition, question-choice, or simulator calls;
- all checklist responses parsed;
- all 18 identical dynamic/fixed paths received identical bit strings;
- zero retries, reasoning tokens, and forced exits;
- cost `$0.00534675`, below the `$0.05` cap.

## Result

- mean dynamic incremental checklist gain: `.2667`;
- mean fixed incremental checklist gain: `.4333`;
- mean dynamic-minus-fixed gain: `-.1667`;
- paired outcomes: `0/25/5` wins/ties/losses;
- positive-mean fixtures: `0/6`;
- all 18 identical-action cells tied exactly;
- among the 12 cells where V3 exact EIG changed the action: `0/7/5`.

The transport, accounting, support-change, candidate-informativeness,
choice-difference, diversity, and identical-path gates passed. The dynamic
gain, dynamic-over-fixed, paired-win, and positive-fixture gates failed.

## Interpretation

V3's fresh paired simulator calls did add noise: three identical-action cells
had differed in its original endpoint. Eliminating that noise makes the causal
diagnosis sharper, not more favorable. Every remaining loss occurs where the
regenerated-support exact-EIG rule chose a different question from the
fixed-support rule, and it produced no corresponding win.

This localizes the failure to the first link. GPT-5.4 can generate changed,
semantically plausible support and answer partitions, but the induced exact-EIG
ranking does not identify questions that reveal more of the official hidden
information.

The public `MECHANICS.json` SHA-256 is
`9855e57d3a00afcd074e15812aa14bb987ede1bc8b09d97a9382ae8b901451f2`.
The private raw-response SHA-256 is
`e5a128d92fa48693e3b6f28b216d3b8c314bb3c67081ad1ba6cdb485e932dc9a`.

This is explicitly post hoc development evidence and cannot support a
confirmatory or headline claim. The exact route is closed without a rerun.
