# Bongard Luna August 10 Current-Commit Readiness

Date checked: 2026-08-08.

Decision: `ready_without_paid_calls`.

The exact production command was run in preflight mode from pushed commit
`a06b538a`. It made zero model calls, wrote no paid execution artifact, and
left authenticated cumulative OpenRouter credits and usage byte-identical:

- total credits: `$245.000000000`;
- total usage: `$220.121013787`;
- live balance: `$24.878986213`.

The reported `$30` top-up remains unposted and is not counted.

## Bound Protocol

- wrapper SHA-256: `b17fcd89...e4ac53`;
- Development64 V13: `1120eef6...e44ddb`;
- naive first-link V3: `ebacf625...a06ec7`;
- Confirmation96 V10: `dce0a42e...16c4c`;
- sample-size expansion V2: `eb4d8284...05d4`.

All independent manifest and execution verifiers passed. The full Bongard
regression family passed `142/142` before this read-only check.

The frozen input replay verified the exact source and image-integrity
manifests, the 5,124,375,111-byte archive, four mechanics tasks, 56 mechanics
images, ten hidden-state-clean serving prompts totaling 8,821,758 serialized
bytes, and the strict belief schema.

## Live Contract

OpenRouter currently exposes exact model `openai/gpt-5.6-luna` with text,
image, and file input; structured output; 1.05M context; and 128K maximum
completion. Live prices remain `$0.10/M` input and `$0.60/M` output. The frozen
`$0.004` request precharge covers the 3,200-token output maximum plus 20,800
prompt tokens.

The wrapper, serving, mechanics, and August 10 ledger paths are all absent.
The account-wide daily cap remains `$5.00`; serving and mechanics component
caps sum to `$2.00`. Unused allowance is not automatically consumed.

Authoritative machine-readable record:

`results/nonmyopic/BONGARD_OPENWORLD_LUNA_AUG10_CURRENT_COMMIT_READINESS_20260808.json`

The only paid command authorized for August 10 remains:

```bash
set -a
source .env
set +a
/opt/anaconda3/bin/python scripts/bongard_openworld_luna_aug10_execute.py --preflight
/opt/anaconda3/bin/python scripts/bongard_openworld_luna_aug10_execute.py
```

The second command is permitted only if the fresh same-day preflight again
returns `ready_without_paid_calls`.
