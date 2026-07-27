# LongVidSearch Four-Hop Support Smoke V2 Launch Amendment

## Status

Frozen after the zero-call V1 launcher failure and before any model response.

## Sole Change

V2 exports variables loaded from `.env` before constructing the adapter:

```bash
set -a
source .env
set +a
```

The committed local launcher
`scripts/run_longvid_four_hop_support_smoke_local.sh` performs exactly this
step and then invokes the frozen Python smoke.

## Unchanged Protocol

Everything scientific and model-facing remains exactly as preregistered:

- `openai/gpt-5.4`, nonreasoning, temperature `0`;
- exact 10 requests and HTTP attempts;
- one discarded preflight, one initial support, eight branch refreshes;
- the same fixture and SHA-256;
- the same six-line pipe grammar;
- raw checkpoint before parse;
- zero retries, repair, reparse, reissue, or partial subset;
- all frozen support/anchor/divergence gates;
- projected cost `$0.08` and hard cap `$0.20`;
- no LongVid scientific task, caption, evidence set, or endpoint.

V1 outputs cannot influence V2 because V1 produced no request or response.
Any V2 failure closes this exact interface.

