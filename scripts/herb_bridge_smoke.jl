using Herb
using TOML

# Lower only data-shaped call expressions; execution stays in the Hodel sandbox.
function graph(expression)
    steps = Any[]
    memo = Dict{Any,String}()
    function lower(x)
        x isa Symbol && return string(x)
        x isa Expr && x.head == :call && x.args[1] isa Symbol ||
            error("unsupported expression")
        haskey(memo, x) && return memo[x]
        args = [lower(a) for a in x.args[2:end]]
        id = "x$(length(steps))"
        push!(steps, Dict("id"=>id, "op"=>string(x.args[1]), "args"=>args))
        memo[x] = id
        return id
    end
    output = lower(expression)
    return Dict("steps"=>steps, "output"=>output)
end

function collect_candidates()
    grammar = @csgrammar begin
        Grid = I
        Grid = identity(Grid)
        Grid = vmirror(Grid)
        Grid = hconcat(Grid, Grid)
        Grid = apply(Callable, Grid)
        Grid = repeat(Row, TWO)
        Grid = paint(Grid, Object)
        Callable = identity
        Callable = compose(Callable, Callable)
        Row = first(Grid)
        Object = asobject(Grid)
    end
    init_probabilities!(grammar)
    iterator = CostBasedBottomUpIterator(grammar, :Grid;
        current_costs=HerbSearch.get_costs(grammar),
        program_to_outputs=nothing, max_depth=5, max_size=7)
    rows = Any[]
    for node in Iterators.take(iterator, 128)
        frozen = freeze_state(node)
        expr = rulenode2expr(frozen, grammar)
        # A bare input is not a legal output reference in the existing graph API.
        expr == :I && (expr = :(identity(I)))
        push!(rows, Dict("expression"=>string(expr), "graph"=>graph(expr)))
    end
    return rows
end

first_branch = collect_candidates()
second_branch = collect_candidates()
@assert first_branch == second_branch
expressions = Set(row["expression"] for row in first_branch)
for required in ("identity(I)", "vmirror(I)", "hconcat(I, I)",
                 "apply(identity, I)", "repeat(first(I), TWO)", "paint(I, asobject(I))")
    @assert required in expressions required
end
TOML.print(stdout, Dict("status"=>"iterator_bridge_pass", "rows"=>first_branch,
    "branch_replay_equal"=>true, "calls"=>0, "benchmark_examples"=>0))
