# Source-only recursive output fidelity

Motivation: opened physics tree calls both returned pi. This cannot distinguish
task response from representation/serving limitations. Official OpenAI documentation
supports recursive schemas in general, not proof of this OpenRouter route:
https://developers.openai.com/api/docs/guides/structured-outputs#supported-schemas

Exactly2 Luna medium calls, same seed57100001,16384tokens. First recursive schema
scalar_tree.schema(2); second nonrecursive schema {payload:string}. Both prompts
explicitly supply the same complete target object, a single tree for
x0 + 2*sin(x1). Recursive arm copies it directly; string arm serializes it in payload.
No physics source, histories, outcomes or data-dependent target. No reasoning about
unknown functions is required. Shape is fixed in tested code before calls.

Cap$.08 total, reserve$.04 per attempt, existing authenticated catalog/credit and
London$5daily ledger checks. No retry or alternative target after any response.
Transport/receipt failures stop. Normal responses are evaluated as format-valid
and exact-copy booleans; semantic/format mismatch is banked and does not suppress
the other arm. All raw responses retained. Exact copying is parsed-object equality,
ignoring whitespace/key order but requiring the requested structure and constants.

Bothpass demonstrates one nontrivial live recursive output, not reliable physics
reasoning or general recursive support. Recursivefail/stringpass implicates this
interface/serving path but does not uniquely identify provider or model fault.
Bothfail leaves cause unresolved. Recursivepass/stringfail validates only recursive
capability on this target. No result rescues closed physics nulls or authorizes
depth. Commit tested code before calls; replay requests/bindings/receipt sum/results.
