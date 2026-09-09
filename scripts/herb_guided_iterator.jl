using Herb
using TOML
include("/input/grammar.jl")
init_probabilities!(grammar)
weights = TOML.parsefile(ARGS[1])["weights"]
@assert length(weights) == length(grammar.log_probabilities)
@assert all(isfinite(w) && w > 0 for w in weights)
@assert isapprox(sum(weights), 1.0)
grammar.log_probabilities .= log.(weights)
rows = String[]
root_grid_only = length(ARGS) > 1 && ARGS[2] == "grid-root"
@assert rulenode2expr(RuleNode(1), grammar) == :I
start = root_grid_only ? Hole(BitVector([i == 1 || !grammar.isterminal[i]
    for i in eachindex(grammar.isterminal)])) : :Value
for candidate in Iterators.take(MLFSIterator(grammar, start; max_depth=5, max_size=12), 64)
    push!(rows, string(rulenode2expr(freeze_state(candidate), grammar)))
end
TOML.print(stdout, Dict("expressions"=>rows, "status"=>"complete", "count"=>length(rows),
    "all_rules_positive"=>true, "root_grid_only"=>root_grid_only,
    "llm_calls"=>0, "search_weights_are_posterior"=>false))
