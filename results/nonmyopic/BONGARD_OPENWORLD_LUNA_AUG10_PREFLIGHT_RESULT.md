# Bongard Luna August 10 Preflight Result

Date checked: 2026-08-06

Decision: `ready_without_paid_calls`.

The read-only preflight made zero model calls and wrote zero files. It verified:

- frozen source commit/tree and metadata hashes;
- source-protocol manifest SHA-256 `7acd3cc9...e763`;
- image-integrity manifest SHA-256 `239943ae...ee96`;
- the banked 5,124,375,111-byte archive binding;
- exactly four mechanics tasks, 56 images, and ten hidden-state-clean serving
  prompts (8,820,158 serialized message bytes);
- strict belief-schema SHA-256 `23239262...9fd`;
- precharge amendment SHA-256 `75acd7ae...bbbff4`;
- interface-v7 development manifest SHA-256 `a649a769...5d9ce1`;
- a four-root, 64 conditioned-branch, 64 paired history-blind-branch mechanics
  tree with 132 first-stage and at most 172 total requests;
- absent wrapper, serving, mechanics, and August 10 ledger artifacts;
- live `openai/gpt-5.6-luna` image and structured-output support, 1.05M context,
  and 128K maximum completion; its `$0.004` attempt reservation covers the
  3,200-token output maximum plus 20,800 prompt tokens at live prices;
- live OpenRouter balance `$27.702109737`, above the `$5` start gate.

Budget boundaries are a `$5.00` account-wide daily cap, `$0.25` serving cap,
and `$1.75` mechanics cap. Mechanics remains conditional on the observed
serving cost projection. The maximum component-cap sum is `$2.00`; unused
allowance is not automatically spent.

The authoritative August 10 command now invokes this same preflight
automatically before any path or ledger write. It requires
`ready_without_paid_calls` and uses the returned live-credit snapshot as the
ledger opening boundary. A failed gate makes no component call and leaves all
target paths absent; a banked ledger or component remains on the existing
no-repeat replay path.

The implementation regression suite passes 92 Bongard tests. The paired
request audit checks exact same-seed, same-batch adjacent dynamic/blind pairs,
initial-history-only blind prompts, and dynamic prompts that add exactly one
simulated answer.
Terminal requests now additionally use one common requested seed per task and
place distinct dynamic/history-blind histories adjacently, eliminating
avoidable per-history seed luck from the primary endpoint contrast. The exact
pairing is persisted and independently replay-gated.
Complete task-level terminal groups are now packed into explicit dispatch
batches of at most 24 requests without splitting a task. Thus distinct
dynamic/history-blind terminal requests share the same adapter invocation as
well as the same requested seed. The dispatch manifest is persisted, and an
adversarial index-23 boundary plus a resized-manifest replay both fail closed.
The rebound development manifest also binds the pre-outcome four-tier claim
classifier; policy-only or mechanism-only outcomes cannot authorize
confirmation.

Command:

```bash
set -a
source .env
set +a
/opt/anaconda3/bin/python \
  scripts/bongard_openworld_luna_aug10_execute.py --preflight
```
