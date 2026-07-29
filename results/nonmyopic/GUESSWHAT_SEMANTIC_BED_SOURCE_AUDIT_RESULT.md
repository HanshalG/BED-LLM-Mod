# GuessWhat?! Semantic BED Source Audit Result

Date: 2026-07-29

Status: **all zero-call source gates pass; multimodal serving smoke
authorized**.

Model calls and cost: `0` / `$0`.

Public manifest:
`results/nonmyopic/guesswhat_semantic_bed_source_audit/guesswhat-semantic-bed-source-audit-20260729/MANIFEST.json`.
SHA-256:
`027a7f49fc599001eca6fbac0b3fe3e6be021d8cb3512eacad3ee167d0f5fff2`.

## Released Source

The audit binds the official GuessWhat?! repository at commit
`346b7de65d5f18fb8c7d357b7c743d02be429d8a` and the official compressed test
archive at SHA-256
`c26c08fbb860786f25ab6940dab135c4f61a6404d0c89bbf5b7a21716306548c`.

The archive contains exactly 23,115 games. Every released target object ID is
present among its row's object annotations.

## Eligibility

Exactly 6,566 rows pass every semantic-game requirement:

- successful released human game;
- 5 through 12 candidate objects;
- at least four valid human question-answer pairs;
- valid non-crowd target and object annotations; and
- at least one repeated object category.

After retaining only one hash-first dialogue per image, 5,931 unique eligible
images remain, above the frozen minimum of 5,000. Eligible object counts cover
the full 5--12 range.

The repeated-category criterion matters because it prevents category naming
alone from identifying every target. A useful policy must distinguish at
least some objects through visual attributes, relations, or position.

## Frozen Split

Seed `39000` produces image-disjoint splits:

| Split | Games/images |
|---|---:|
| Serving smoke | 2 |
| Development | 20 |
| Holdout | 60 |
| Unused | 5,849 |

The two serving games contain seven and six candidate objects respectively,
and both have same-category ambiguity.

The public manifest includes only source dialogue/image IDs, full-row hashes,
image filenames/URLs, and structural counts. It excludes target IDs, human
questions and answers, game outcomes, and object categories.

## BED Adaptation

The planned policy sees the source image with numbered candidate boxes and
its own past questions and realized answers. It does not see the released
target, human dialogue, outcome, or category annotations.

A planning model will propose natural-language yes/no questions. A separate
visual model will estimate answer probabilities over all candidate boxes, and
an independent visual oracle will answer for the released target box. Exact
target-object identification is the terminal endpoint.

This is a new visual-semantic BED adaptation, not a claim about the released
legacy TensorFlow models.

## Decision

All nine conjunctive source gates pass. This authorizes only a separately
frozen exact ten-call multimodal serving smoke on the two serving images.
Development and holdout endpoints remain unopened.
