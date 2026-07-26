# InfoQuest Refresh-Continuation Mechanics Result

The single authorized disclosed-ID mechanics run failed closed at the frozen
fixed-support parser. No scientific endpoint was evaluated.

## Failure

- run ID:
  `infoquest-refresh-mechanics-20260726T025048Z`;
- exact 93 physical requests and 93 HTTP attempts completed:
  3 initial-support, 30 root-simulator, 30 dynamic-refresh, and 30
  fixed-support calls;
- zero retries, reasoning tokens, and forced exits;
- cost `$0.3897848`;
- all four completed batches were checkpointed before parsing;
- the first invalid fixed branch was flat index 15, disclosed record `1`,
  world `2`, root index `0`;
- its follow-up was
  `Are you focusing on adolescents, adults, or early childhood?`;
- this contains `or` and therefore violates the preregistered atomic-question
  grammar.

The run stopped before the 60 paired follow-up-simulator calls and all six
official-checklist calls. Consequently, it emitted no support-change aggregate,
dynamic-versus-fixed endpoint, gate verdict, or policy-efficacy claim.

The public `GATE_FAILURE.json` SHA-256 is
`35bcb52649032ad97333472bfaa396aa5f261f164289def65576ff776f00a242`.
The checkpointed private raw-response SHA-256 is
`d6019ffd6fa1ac782a30c101736783bcad69215e0a3c56808d80f8c4ac4c5c36`.

No response was repaired, reparsed under a relaxed grammar, replaced, or
reissued. This exact refresh-continuation route is closed. The passed serving
gate remains only an interface result.
