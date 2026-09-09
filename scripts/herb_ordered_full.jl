using Herb
using TOML
include("/app/herb_ordered_iterator.jl")
include("/input/grammar.jl")
results = Dict()
for mode in ("base", "guided")
    branch = deepcopy(grammar)
    init_probabilities!(branch)
    weights = TOML.parsefile("/guidance/$(mode).toml")["weights"]
    @assert length(weights) == length(branch.log_probabilities)
    @assert all(w -> isfinite(w) && w > 0, weights) && isapprox(sum(weights), 1.)
    branch.log_probabilities .= log.(weights)
    @assert rulenode2expr(RuleNode(1), branch) == :I
    root = Hole(BitVector([i == 1 || !branch.isterminal[i] for i in eachindex(branch.isterminal)]))
    iterator = SingletonMLFSIterator(branch, root; max_depth=5, max_size=12, max_expansions=50000)
    expressions, scores = String[], Float64[]
    started = time()
    for candidate in Iterators.take(iterator, 64)
        frozen = freeze_state(candidate)
        push!(expressions, string(rulenode2expr(frozen, branch)))
        push!(scores, max_rulenode_log_probability(frozen, branch))
    end
    @assert all(diff(scores) .<= 1e-12)
    results[mode] = Dict("expressions"=>expressions, "log_weights"=>scores,
        "root_grid_only"=>true, "count"=>length(scores), "expansions"=>iterator.expansions[],
        "seconds"=>time()-started)
end
TOML.print(stdout, Dict("status"=>"full_order_pass", "arms"=>results, "calls"=>0))
