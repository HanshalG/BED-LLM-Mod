# InfoQuest Discrete-Action Mechanics Result

The preregistered shared-action mechanics run completed cleanly but failed its
scientific gates. Regenerating support changed the LLM's action in some cells
and never improved the official checklist endpoint.

## Execution

- run ID: `infoquest-discrete-mechanics-20260726T030149Z`;
- exact 159 physical requests and 159 HTTP attempts;
- zero retries, reasoning tokens, and forced exits;
- cost `$0.44610855`, below the `$0.85` cap;
- all response batches parsed and were checkpointed before parsing;
- fixture SHA-256
  `63248345976e884b9fac127033415fe68bacb3a0e32a32e498285851719251cc`;
- no opportunity, development, or holdout record was read.

## Frozen Metrics

Support regeneration itself was strong:

- 3/3 distinct initial-support hashes;
- 30/30 dynamic supports changed;
- mean dynamic novel-support fraction `1.0`;
- all 6 fixtures used at least two dynamic choice labels;
- dynamic endpoint range was at least one on 5/6 fixtures.

The causal action and endpoint gates failed:

- dynamic and fixed choices differed on `12/30`, below `20/30`;
- dynamic next-turn checklist gain was `0.1667`, below `0.50`;
- fixed next-turn checklist gain was `0.3333`;
- dynamic minus fixed was `-0.1667`, below `+0.15`;
- paired dynamic versus fixed outcomes were `0/26/4` wins/ties/losses;
- `0/6` fixtures had positive mean dynamic-minus-fixed gain.

The public `MECHANICS.json` SHA-256 is
`140f77447da9bd3d83e8a7be7fd45dc8fc792422d4e1cef398f6d8460d13ccc3`.
The checkpointed private raw-response SHA-256 is
`c4dec013386083afb4279754f4ffb1b0a2f6559bb2a2cf1863d80a849fe6e93e`.

## Interpretation

This is adverse first-link evidence, not a transport failure. GPT-5.4 replaced
every dynamic support with entirely novel hypotheses and made diverse choices,
but those choices produced no checklist win and four losses against the
compute-matched fixed-support branch. Wholesale path-dependent regeneration was
therefore not a calibrated value signal in these disclosed InfoQuest mechanics
fixtures.

The exact shared-action route is closed. It does not authorize root-ranking,
policy, development, holdout, or depth experiments.
