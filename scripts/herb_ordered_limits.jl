using Herb
using Test
using TOML
include("/app/herb_ordered_iterator.jl")
grammar = @csgrammar begin
    Value = x
    Value = y
    Value = f(Value)
end
init_probabilities!(grammar)
a = SingletonMLFSIterator(grammar, :Value; max_depth=3, max_expansions=1)
b = SingletonMLFSIterator(grammar, :Value; max_depth=3)
@test a.expansions !== b.expansions
@test_throws ErrorException collect(a)
@test a.expansions[] == 1
@test b.expansions[] == 0
grammar.log_probabilities[1] = .1
@test_throws ErrorException collect(b)
@test b.expansions[] == 0
TOML.print(stdout, Dict("status"=>"limits_pass", "cap_preserved"=>true,
    "invalid_weights_rejected"=>true, "independent_counters"=>true, "calls"=>0))
