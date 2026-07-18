# COPEx Direct-Proposal DeepSeek Quality-Screen Failure

The DeepSeek V4 Flash direct-angle quality screen was stopped before it completed the
eight fixed states. This is a tractability/interface failure, not a proposal-quality
or policy result.

The first attempt returned a 20-token nonconforming response. Its one permitted repair
eventually returned after several minutes and used high reasoning. The next state then
began another multi-minute request. At that point the process was terminated to avoid
spending the remaining screen budget on an endpoint that cannot practically supply
thousands of simulated-child proposal cells for d2 planning.

The spend ledger recorded three completed requests before termination: `$0.00377837`,
11,806 completion tokens, and 10,940 reasoning tokens. No complete score artifact or
policy trajectory exists, so this route does not inform candidate quality or any
non-myopic comparison. The previously observed strategy-JSON serving success does not
transfer to the small direct-angle JSON interface at adequate latency.
