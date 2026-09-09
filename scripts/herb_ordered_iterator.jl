# Reuse Herb's solver/queue, but do not emit unresolved uniform-shape batches.
using Herb

@programiterator SingletonMLFSIterator(
    expansions::Base.RefValue{Int}=Ref(0),
    max_expansions::Int=50000,
) <: TopDownIterator

function HerbSearch.priority_function(::SingletonMLFSIterator, grammar::AbstractGrammar,
    tree::AbstractRuleNode, parent::Union{Real,Tuple{Vararg{Real}}}, requeued::Bool)
    all(x -> isfinite(x) && x <= 0, grammar.log_probabilities) || error("invalid_log_weights")
    -max_rulenode_log_probability(tree, grammar)
end

function HerbSearch.hole_heuristic(::SingletonMLFSIterator, tree::AbstractRuleNode, max_depth::Int)
    function visit(node, remaining, path)
        remaining <= 0 && return HerbSearch.LimitReached()
        !isfilled(node) && return HoleReference(node, path)
        for (index, child) in enumerate(node.children)
            result = visit(child, remaining-1, [path; index])
            result isa HerbSearch.AlreadyComplete || return result
        end
        HerbSearch.AlreadyComplete()
    end
    visit(tree, max_depth, Int[])
end

function HerbSearch._decide_hole(solver::Solver, queue::HerbSearch.PriorityQueue, iter::SingletonMLFSIterator,
    ::SolverState, parent_priority, reference::HoleReference)
    indices = findall(reference.hole.domain)
    width = length(reference.hole.domain)
    for index in indices
        iter.expansions[] >= iter.max_expansions && error("expansion_cap")
        iter.expansions[] += 1
        saved = save_state!(solver)
        mask = falses(width)
        mask[index] = true
        remove_all_but!(solver, reference.path, mask)
        if isfeasible(solver)
            priority = -max_rulenode_log_probability(get_tree(solver), get_grammar(solver))
            push!(queue, get_state(solver) => priority)
        end
        load_state!(solver, saved)
    end
    nothing
end
