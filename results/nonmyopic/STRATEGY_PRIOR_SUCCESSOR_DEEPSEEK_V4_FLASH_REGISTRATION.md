# Strategy-Prior Successor: DeepSeek V4 Flash Serving Registration

Registered on 2026-07-18 before any V4 Flash model call. The previous V4 Pro gate did
not reach final JSON because its catalog-supported reasoning levels are high/xhigh, but
the attempted medium-reasoning 2,048-token configuration did not reserve enough output
capacity. V4 Pro cannot receive the larger viable allocation within the successor's
`$3.00` endpoint cap.

`deepseek/deepseek-v4-flash` is selected because the current OpenRouter catalog lists
native `high` reasoning and a `$0.196/M` completion price. Its 8,192-token total-output
configuration keeps high reasoning enabled while leaving substantial final JSON capacity.
The mandatory gate is unchanged: five L1 plus five L3 strict first-response cells, all
parsed and executed with no repair and zero forced exits.

The frozen formal L3 endpoint is unchanged. At 1,373 reference calls, an 8,192-token
maximum costs approximately `$2.20` in completion tokens; historical prompts add about
`$0.05` at `$0.098/M`. The configuration projects `$2.30` and is independently hard
capped at `$3.00`. No endpoint is permitted unless this serving gate passes.
