# InfoQuest Cached-Partition EIG Mechanics Result

The frozen disclosed-world mechanics run failed closed during dynamic semantic
partition parsing. Efficacy was not measured.

## Result

- run ID: `infoquest-cached-partition-mechanics-20260726T032406Z`;
- all 30 dynamic-partition requests completed and were checkpointed;
- exact 30 physical requests and 30 HTTP attempts;
- zero retries, reasoning tokens, and forced exits;
- cost `$0.2460975`;
- dynamic cell 3 emitted `b5=4`, outside the frozen `0–3`
  answer-cluster alphabet;
- no fixed-partition, simulator, checklist, or scientific endpoint call ran;
- no response was repaired, reparsed, reissued, or replaced.

A post-failure transport-only audit of the already checkpointed responses found
that 5 of 30 dynamic cells used labels from `4` through `7`. The model was
systematically assigning additional unique cluster identifiers rather than
respecting the four-label interface; this was not an isolated serialization
typo.

The public `GATE_FAILURE.json` SHA-256 is
`35bb34ecdb84034c3eb1a86063d58d8c10dd5cc3585a0f2da91b2d206cf45855`.
The private raw-response SHA-256 is
`788a5cb62d61884553943fe4186858c6ceebd95b53b77fd9a42fab3e43387a7e`.

Under the preregistered no-repair and no-rerun rule, this exact cached-history
semantic-partition route is closed. It provides interface diagnosis only and
neither positive nor negative evidence about non-myopic policy efficacy.
