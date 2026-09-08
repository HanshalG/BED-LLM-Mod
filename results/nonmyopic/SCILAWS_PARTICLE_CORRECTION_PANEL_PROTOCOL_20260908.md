# Full linear-correction qualification

Freeze the full48-case public panel before execution: eight ordered tasks,
zero/affine/quadratic histories, seeds1304/1305. Unchanged512 Sobol particles per
family, all8 actions,64 targets, particle-specific observation and target noise.
Evaluate one-step correction at4/8/16/32 branches, all counts on all cases.

Reuse exactly the three correction-workload cases, SHA256
49ccf51cb49f94bd7fd996013519b89859d28cbcfe04618531b99a0bdd0d772a.
Reuse all independent reference values from the full integration panel, SHA256
06f7f6c21ceda8a34ee5cc96f7199d3da677179db241d9c6b1f20e5288e0353d.
Do not repeat either banked computation. Forty-five new correction cases only;
exclusive directory/shards, preserve any failure prefix, no retry or overwrite.

Each count/case shares5s/100000states/64MiB. Every root absolute error and
reference action regret <=1e-4, references error+tail <=1e-7 and mass error<=1e-8,
finite values and full exact coverage required. All48 cases must pass to qualify
a count, independently recomputed rather than trusting saved passed flags.
No gate changes or subset rescue. All outcomes, including nulls, are banked.

A pass qualifies only initial-history one-step integration. Simulated future
histories and genuine full-horizon Bellman values remain separate requirements.
No source measurements, paid requests, LLM interface, or deep policy experiment
authorized by this protocol. Leave daily allowance unused and automation paused.
