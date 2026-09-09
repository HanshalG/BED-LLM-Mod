using Herb
using TOML
using SHA
include("/app/herb_ordered_iterator.jl")
include("/input/grammar.jl")
request = TOML.parsefile("/input/request.toml")
weights = request["weights"]
count_limit, expansion_limit = request["count"], request["max_expansions"]
@assert (count_limit, expansion_limit) in ((56,50000), (128,100000))
init_probabilities!(grammar)
@assert length(weights) == length(grammar.log_probabilities)
@assert all(w -> isfinite(w) && w > 0, weights) && isapprox(sum(weights), 1.)
grammar.log_probabilities .= log.(weights)
@assert rulenode2expr(RuleNode(1), grammar) == :I
root = Hole(BitVector([i == 1 || !grammar.isterminal[i] for i in eachindex(grammar.isterminal)]))
iterator = SingletonMLFSIterator(grammar, root; max_depth=5, max_size=12, max_expansions=expansion_limit)
expressions, scores = String[], Float64[]
status, failure = "complete", ""
try
    for candidate in Iterators.take(iterator, count_limit)
        frozen = freeze_state(candidate)
        push!(expressions, string(rulenode2expr(frozen, grammar)))
        push!(scores, max_rulenode_log_probability(frozen, grammar))
    end
    @assert length(expressions) == count_limit && all(diff(scores) .<= 1e-12)
catch error
    status, failure = "failed", sprint(showerror, error)
end
TOML.print(stdout, Dict("status"=>status, "failure"=>failure, "expressions"=>expressions,
    "log_weights"=>scores, "expansions"=>iterator.expansions[],
    "manifest_sha256"=>bytes2hex(sha256(read("/work/env/Manifest.toml"))),
    "no_api_key"=>!haskey(ENV,"OPENROUTER_API_KEY")))
