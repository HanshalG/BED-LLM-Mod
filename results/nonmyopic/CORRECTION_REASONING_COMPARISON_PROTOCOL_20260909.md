# New-source medium/high reasoning qualification

Frozen model-capability diagnostic, not a BED result or old-source retry. Use four
new synthetic source laws in order:
base x0, source x0*(1+.3*x1);
base sqrt(x0), source sqrt(x0)/(1+.4*x1);
base x0, source x0+.2*exp(x1);
base x0/(1+x0), source base*(1+.5*cos(x1)).
Sources are positive on [.5,2]^2 and never enter prompts. Same six observation
inputs as prior qualification, log-noiseSD.05, new noise seeds60100000+i, target
input seeds60200000+i,32independent log-uniform targets. Consume6history noise
draws before32target noise draws. No source selections or gate changes after data.

Full2x2 design: medium/high effort x old3/new6-observation proposer,4sources=16calls.
Shared paired seed60300000+i,16384token cap, same model openai/gpt-5.6-luna,
OpenAI-only route. Only effort differs between medium/high requests. Both use the
unchanged compositional correction schema/prompt,8candidate cap and numerical
updater. Both final predictors use all6observations and the same calibrated base.
Cases even: medium then high, control then refresh. Cases odd: high then medium,
refresh then control. No cross-arm outputs. Numerical ridge comparator unchanged.

Apply the previous qualification thresholds independently per effort: all defined,
new-history meanMSE <=.9old-history mean, >=2paired improvements>.01, nonworse ridge.
Report all per-case losses, base/ridge comparators, differences, effort token/cost
totals and both gates regardless of outcome. High beating medium alone cannot
qualify it. Any pass only motivates fresh source transfer; neither source simplicity
nor finite grammar establishes LLM necessity. No depth authorization.16calls cannot
establish a general reasoning superiority or statistical significance.

Whole block cap$.64, each request reserves$.04 under the account-wide London$5cap.
Live route/credits before attempts. Existing16384cap is unchanged: this tests effort,
not a higher token ceiling. Unsupported high/schema/transport/decoding fails closed,
no retries or fallback. Protocol/tested executor committed before calls. All16
forecasts sealed before targets are evaluated. Replay histories, requests, receipts,
all bound implementations, endpoints and gates. Old qualification remains closed.
