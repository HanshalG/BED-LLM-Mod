# InfoQuest Trajectory Access Amendment

Recorded immediately after the first cached-trajectory mechanics inspection
and before reading any opportunity trajectory or defining opportunity
endpoints.

## Incident

The source manifest correctly validates `seed_messages.jsonl`,
`settings.jsonl`, and `traits.jsonl` as ID-ordered. The three released Falcon
30-turn baseline JSONL files are different: each contains all 500 unique IDs,
but rows are not ordered by ID.

The first mechanics-only inspection incorrectly selected row positions `0` and
`1` instead of mapping rows by their `id` field. Across the three baseline
files, those positions correspond to IDs `0` and `4`. Cached trajectory text
for ID `4` was therefore exposed even though ID `4` belonged to the frozen
holdout. No opportunity or development ID was exposed.

Source-setting content previously disclosed during source-shape inspection
remains IDs `0` and `1`. The union of disclosed IDs is therefore `{0, 1, 4}`.

## Prospective Quarantine

1. ID `4` is permanently excluded from all opportunity, development, holdout,
   tuning, ranking, policy, and confirmation evidence.
2. ID `4` may be used only as disclosed mechanics material or ignored.
3. Every cached-trajectory loader must validate 500 unique IDs with exact set
   `0..499` and select records through an explicit `id -> row` map.
4. The frozen opportunity 80 and development 30 remain unchanged.
5. The effective holdout is the original holdout with ID `4` removed, leaving
   387 records.

Effective holdout ordered SHA-256:
`5e32d2ef9567c67794ba0cecf3cacf5a40429deeae1a80384c1aaf8101de085b`.

Effective combined-splits SHA-256, retaining the original mechanics,
opportunity, and development arrays and removing ID `4` from holdout:
`2e8fb9e9a328a99258243fbf39bd9ad057899ebbfa342068bf7bd5d8e40cc827`.

The opportunity hash remains
`737296c449680cfb7c78aa03d44508af487eb9624d0e416d37cf5ddbaa01958b`;
the development hash remains
`068587d494c71d5488da4f1d53ebe34be713c178083f6be98733085337c48b48`.

This amendment does not authorize opportunity access. Opportunity metrics and
thresholds must still be implemented, tested on disclosed IDs only, and
committed before loading any of the 80 opportunity trajectories.

OpenRouter calls/cost: `0 / $0`. OatML jobs: `0`.
