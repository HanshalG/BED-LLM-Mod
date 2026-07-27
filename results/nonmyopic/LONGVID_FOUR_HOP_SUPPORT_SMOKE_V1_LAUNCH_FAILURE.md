# LongVidSearch Four-Hop Support Smoke V1 Launch Failure

## Decision

The first live launch made zero API calls and produced no model response. V1 is
closed as launched; it is not counted as a serving result.

Run ID:
`longvid-four-hop-support-smoke-20260727T145010Z`.

## Failure

The shell command used:

```bash
source .env
python scripts/longvid_four_hop_support_smoke.py ...
```

The `.env` assignment became a shell variable but was not exported to the
Python process. `OpenRouterAdapter` therefore raised:

`ValueError: OPENROUTER_API_KEY is required for backend=openrouter`

during construction.

The failure occurred before the script's model-execution `try` block, so no
public or private response artifact was written.

## Accounting

- physical requests: `0`;
- HTTP attempts: `0`;
- responses: `0`;
- prompt/completion/reasoning tokens: `0/0/0`;
- cost: `$0`;
- LongVid task/caption/endpoint access: `0`.

An authenticated post-failure credit check remained exactly:

- total credits: `$140`;
- usage: `$106.425957406`;
- remaining: `$33.574042594`.

No model, prompt, parser, gate, fixture, or budget outcome was observed.

