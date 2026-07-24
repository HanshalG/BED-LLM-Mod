# Countries CA-BED Aligned Global-Bank V2 Result

Date: 2026-07-24

## Decision

The final Countries serving smoke failed its frozen bank-completeness gate. No
depth score, target endpoint, or formal response was produced. Countries is
closed without a V3, retry, row drop, threshold change, or post-hoc endpoint.

## Integrity and Cost

Run `countries-cabed-aligned-bank-v2-smoke-20260724T222004Z` completed:

- exactly 82 physical requests: 2 full GPT-5.4 question-bank requests and 80
  GPT-5.4 Mini semantic-table requests;
- zero reasoning tokens, retries, forced exits, or transport errors;
- 76,845 prompt and 45,783 completion tokens;
- cost `$0.27411700`; and
- both private raw phases checkpointed under SHA-256
  `912a46e0cbc1a8a4f798877f2e174df6ed4996f616cd862eef56e95a89524274`.

Both generated banks and all 80 complete 64-country binary tables parsed.
Tree 0 had 36 rows satisfying the preregistered minimum of four Yes and four No
labels. Tree 1 had 31, one fewer than the required 32.

## Failure Characterization

The invalid rows were all rare or unsupported properties under the model's own
semantic table. Tree 0 had four invalid rows. Tree 1 had nine, including Arabic
or Portuguese official-language questions, former Yugoslav or Soviet
membership, Red Sea and United States borders, the Equator, Scandinavian
Peninsula membership, and a Buddhist-majority question. Their minority label
counts ranged from zero to three.

This is a serving and semantic constraint-following failure. It is not evidence
for or against the non-myopic depth hypothesis. The protocol explicitly
required 32 balanced rows, so computing an endpoint on the available 31-row
tree would be an unregistered threshold repair.

## Consequence

The global bank removed V1's fragile branch-menu generation and made every
completed semantic response auditable, but natural-language generation still
did not reliably honor a cardinality constraint that depends on the subsequent
LLM label table. Per preregistration, there is no Countries V3 and no formal
run.
