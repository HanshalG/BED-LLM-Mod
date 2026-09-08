# Constrained request: metadata eligibility, not a serving pass

The prior typed-continuation work is preserved. This turn adds the transport-free
request builder and tests the missing hosted-route/input-accounting contract.
There are zero model calls and zero new scientific outcomes.

Authenticated OpenRouter endpoint metadata for the exact dated DeepSeek model
advertises structured_outputs, response_format, reasoning, seed, temperature and
max_tokens at OpenInference `open-inference/fp8`. Observed input/output prices
are $0.05/$0.16 per million tokens, below the proposed $0.10/$0.20 ceilings.
The source URL, timestamp and relevant endpoint fields are banked in the adjacent
JSON artifact. This metadata DOES NOT verify support or enforcement for our
particular nested schema. Only a prospectively authorized serving gate could.

The new request builder pins this provider, disables fallbacks and reasoning,
requires parameter support, and requests strict json_schema output. It reuses
only validated public operation descriptions and real histories from the old
interface, not its incompatible `steps` response instructions. Aware and blind
use identical schema/instructions; blind erases history without leaking its size.
No field accepts hidden programs, target outcomes or task seeds.

The full constructed-fixture request is22,966 bytes, including schema and all
parameters. A32,768-byte complete-request limit replaces the old message-only
size assumption for this NEW builder. A65,536-input-token reservation and4096
output cap at the price ceilings give$0.0073728 token exposure, covered by the
proposed$0.015 per-attempt reservation. This does not authorize requests, certify
provider token accounting, or modify the old runner. A future runner must verify
fresh route metadata, actual token/cost reports and daily reservations.

36 constrained-request/language/proposal tests pass in0.65 seconds; lint passes.
Tests cover all typed-language checks from the preceding turn, blind isolation,
complete-request size, price/context/output/parameter/identity rejection, and
explicit false fields for paid authorization and actual schema-serving support.

Next decision: approval for a NEW <=$0.25 constrained-interface proposal-quality
gate, with fresh frozen cases and the aware/blind/symbolic controls. This is not
a retry of the closed response or inheritance of its consumed authorization.
Before dispatch it still needs its prospective case manifest, runner binding,
end-to-end zero-call tests and full accounting checks. Metadata alone authorizes
nothing. No depth experiment, clinical significance or publication claim follows.

LondonSept8 live credits/usage/balance245/220.376763864/24.623236136; daily spend
$0.00006987 remains unchanged. Automation paused; full research goal incomplete.
