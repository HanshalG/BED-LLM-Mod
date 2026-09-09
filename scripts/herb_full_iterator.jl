using Herb
using TOML
include("/input/grammar.jl")

function lower_graph(expression)
    steps = Any[]
    memo = Dict{Any,String}()
    function lower(x)
        x isa Symbol && return string(x)
        x isa Expr && x.head == :call && x.args[1] isa Symbol || error("bad tree")
        haskey(memo, x) && return memo[x]
        name = string(x.args[1])
        args = x.args[2:end]
        if startswith(name, "__bed_call")
            op = lower(args[1])
            args = args[2:end]
        else
            op = name
        end
        refs = [lower(a) for a in args]
        id = "x$(length(steps))"
        push!(steps, Dict("id"=>id, "op"=>op, "args"=>refs))
        memo[x] = id
        id
    end
    # Keep non-grid candidates as explicit execution failures, not silently pruned.
    expression isa Symbol && (expression = Expr(:call, :identity, expression))
    output = lower(expression)
    Dict("steps"=>steps, "output"=>output)
end

rows = Any[]
started = time()
# BFS is lazy; a full bottom-up product can exhaust memory before yielding a call.
iterator = BFSIterator(grammar, :Value; max_depth=5, max_size=12)
for candidate in Iterators.take(iterator, 256)
    expression = rulenode2expr(freeze_state(candidate), grammar)
    push!(rows, Dict("expression"=>string(expression), "graph"=>lower_graph(expression)))
end
fixture = lower_graph(:(__bed_call1(compose(identity, identity), I)))
TOML.print(stdout, Dict("status"=>"bounded_enumeration_complete", "rows"=>rows,
    "computed_callable_fixture"=>fixture, "seconds"=>time()-started,
    "calls"=>0, "benchmark_examples"=>0))
