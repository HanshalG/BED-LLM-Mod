# Failed-Closed Pilot Interface Audit

## Scope

This audit covers only the failed-closed launch
`nonmyopic-ucizoo-llm-pilot-20260714`. It contains no policy metric and is not
scientific evidence for any arm.

## Recorded Failure

The candidate parser required a bare JSON object. After 68 requests and `$0.00271641`,
the run stopped at trial 1 after exhausting the configured two attempts for one root
candidate cell. The raw failure record is `PILOT_FAILURE.json`.

## Exact Wrapper Check

All 10 rejected responses had precisely this form:

~~~~text
```json
{"trait_ids":["legal_id_1","legal_id_2","legal_id_3"]}
```
~~~~

For every one of the 10 records, stripping only the opening ` ```json` line and final
` ``` ` line yielded a JSON object with exactly the `trait_ids` key, exactly three
distinct IDs, and IDs that were legal and unasked in that recorded history. There was
no prose, additional field, unknown trait, duplicate, or re-asked trait in this set.

## Pending Decision

The proposed interface amendment is deliberately narrow: accept a response only when
it is either a bare JSON object or a *single exact* ` ```json ... ``` ` wrapper around
that object. The existing schema, cardinality, legality, and unasked-action checks
would remain unchanged; prose, any other fence language, multiple blocks, and malformed
inner JSON would still fail closed. No amended parser or fresh launch is authorized or
executed by this audit.
