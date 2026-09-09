using Herb
using TOML
include("/app/herb_ordered_iterator.jl")

function audit(grammar, probabilities, depth_limit, size_limit)
    init_probabilities!(grammar)
    grammar.log_probabilities .= log.(probabilities)
    exhaustive = [freeze_state(p) for p in BFSIterator(grammar, :Value;
        max_depth=depth_limit, max_size=size_limit)]
    iterator = SingletonMLFSIterator(grammar, :Value;
        max_depth=depth_limit, max_size=size_limit)
    actual = [freeze_state(p) for p in iterator]
    key(p) = string(rulenode2expr(p, grammar))
    score(p) = max_rulenode_log_probability(p, grammar)
    expected = sort(score.(exhaustive), rev=true)
    @assert Set(key.(actual)) == Set(key.(exhaustive))
    @assert length(actual) == length(exhaustive) == length(Set(key.(actual)))
    @assert all(isapprox(a, b; atol=1e-12) for (a,b) in zip(score.(actual), expected))
    old = [freeze_state(p) for p in MLFSIterator(grammar, :Value;
        max_depth=depth_limit, max_size=size_limit)]
    old_scores = score.(old)
    Dict("count"=>length(actual), "expansions"=>iterator.expansions[],
        "old_order_violations"=>count(diff(old_scores) .> 1e-12),
        "ordered_scores"=>score.(actual), "ordered_expressions"=>key.(actual))
end

unary = @csgrammar begin
    Value = x
    Value = y
    Value = f(Value)
end
binary = @csgrammar begin
    Value = x
    Value = y
    Value = pair(Value, Value)
end
mixed = @csgrammar begin
    Value = x
    Value = y
    Value = f(Value)
    Value = pair(Value, Value)
end
results = [audit(unary, [.6,.1,.3], 4, 7),
           audit(binary, [.6,.1,.3], 3, 7),
           audit(mixed, [.45,.15,.25,.15], 3, 7),
           audit(mixed, [.25,.25,.25,.25], 3, 7)]
TOML.print(stdout, Dict("status"=>"exhaustive_order_pass", "cases"=>results,
    "calls"=>0, "benchmark_examples"=>0))
