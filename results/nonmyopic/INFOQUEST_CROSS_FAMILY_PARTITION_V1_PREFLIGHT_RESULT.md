# InfoQuest Cross-Family Partition V1 Preflight Result

The V1 serving invocation failed closed during local budget preflight before
any OpenRouter request or scientific access.

- run ID: `infoquest-cross-family-serving-20260726T035153Z`;
- projected run spend was `$0.08`, above the frozen `$0.03` serving cap;
- zero physical requests and HTTP attempts;
- zero prompt, completion, or reasoning tokens;
- cost `$0`;
- no prompt, model response, private raw artifact, fixture, or endpoint.

The public failure SHA-256 is
`4d64410acf12d24adca9bd25867d9867a5cdbef9d643d18cb68865cdd8191264`.

V1 is closed. The separately preregistered V2 amendment changes only the local
serving projected-spend hint to `$0.02`; all scientific settings remain frozen.
