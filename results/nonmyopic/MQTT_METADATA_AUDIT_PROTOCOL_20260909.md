# Complete-family MQTT metadata audit

Before reading any MQTT transition-table bytes, freeze all32DOT paths under
eval/src/main/resources/mqtt at mut-learn commit24b5535f37ca92745ddb4c3a5a5381b6ccfa87ce.
No transition-dependent choice, replacement or model-performance selection.

Use Graphviz13.0.1 to parse DOT as data. Validate unique explicit initial state,
deterministic total transitions, declared-state references, reachability and
input/output vocabularies. Report every invalid file; never silently complete
missing transitions or remap outputs. Group valid models by literal scenario
basename AND exact input alphabet. Preserve all groups, including singletons.
No aliasing mosquitto/mosquitto to another scenario based on its behavior.

Only local parser code may see transition bytes. Persist source hashes and
aggregate metadata, not upstream transition tables. No simulator trajectories,
model calls, hypothesis evaluation or planning scores. This is source admission
only. Original repository's GPL3license is distinct from AutomataWiki's stated
MITterms; do not redistribute upstream code/data as MIT on that basis.

After this audit, freeze any numerical opportunity study separately. Source
models in an explicit finite prior may support oracle mechanics, never an
LLM-native claim. End-to-end learning must hide candidate/true machines and
compare against strong automata induction under common physical-step accounting.
Any family/version split must acknowledge shared implementations and scenarios.
