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
- corrected development manifest SHA-256 `d5e8412f...b9c`;
- absent wrapper, serving, mechanics, and August 10 ledger artifacts;
- live `openai/gpt-5.6-luna` image and structured-output support, 1.05M context,
  and 128K maximum completion;
- live OpenRouter balance `$27.702109737`, above the `$5` start gate.

Budget boundaries are a `$5.00` account-wide daily cap, `$0.25` serving cap,
and `$1.75` mechanics cap. Mechanics remains conditional on the observed
serving cost projection. The maximum component-cap sum is `$2.00`; unused
allowance is not automatically spent.

Command:

```bash
set -a; source .env; set +a
/opt/anaconda3/bin/python \
  scripts/bongard_openworld_luna_aug10_execute.py --preflight
```
