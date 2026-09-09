# MQTT full-family metadata: structurally usable, efficacy untested

Manifest74327b2f froze all32files before transition reads. A synthetic parser test
caught Graphviz's empty-label field on unlabelled start edges; correction a084b26f
was tested/pushed before the single source audit. All3tests pass. No scientific
selection changed. Metadata SHA256b2ddd4eb02218334f6d00331e87b0b8ca9652c483f4452873e4b4b2479598e62.

| Literal scenario | Models | Inputs | States, min-max | Output vocabulary sizes |
|---|---:|---:|---:|---:|
|invalid|5|11|3-5|6-8|
|mosquitto|1|7|3|7|
|non_clean|5|6|10-12|17-20|
|simple|4|7|3-5|7-8|
|single_client|5|11|8-10|10-13|
|two_client|4|9|9-16|27-48|
|two_client_same_id|3|11|7|13|
|two_client_will_retain|5|9|17-18|18-22|

All32have a unique explicit start, complete deterministic transitions, and every
state reachable. Grouping uses exact input alphabets AND literal scenario names.
The singleton stays recorded; it was not renamed to enlarge another group.
Different output-vocabulary sizes are allowed; outputs remain exact strings.
Counts do not establish behavioral diversity, reset cost, horizon opportunity or
calibration. The dot files are learned abstractions of historical broker versions,
not complete specifications or assurances about current live implementations.

The [original author repository](https://github.com/mtappler/mut-learn/tree/24b5535f37ca92745ddb4c3a5a5381b6ccfa87ce)
provides the MQTT models and declares GPL3. We persist hashes/derived metadata,
not original model tables or Java code. Its loader delegates pre/post/step to
LearnLib's simulator. A prospective experiment must declare a simulator reset-cost
convention, not imply that a chosen unit cost was measured on real brokers.

This is a source pass, not a sealed ingestion claim: transition bytes were parsed
locally for structure and vocabularies. No command trajectory was generated,
candidate model shown to an LLM or policy risk scored. All32files remain in the
record with no replacement. Source version/scenario overlap precludes treating
them as32independent evaluation tasks.

Next is a separately frozen finite-prior oracle opportunity study of all8groups,
including the singleton. This can cheaply reject zero-headroom cases, but cannot
establish LLM-native discovery because its true-machine set is supplied to the
oracle. A later LLM experiment requires independently specified unseen-model
prediction and a strong equal-budget classical learner. No model calls authorized.

Prior/current turns made source progress. No API cost; authenticated account
unchanged at23:47London, conservative dayremaining3.10986396. No cluster/automation
actions, no old endpoint or gate altered. Goal remains unmet.
