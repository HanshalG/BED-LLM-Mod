"""Bounded first-order grid DSL baseline; not exhaustive full-DSL synthesis."""
import ast
import itertools
from scripts.rearc_graph_worker import grid


def library(source):
    constants = {}
    operations = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError,TypeError):
                continue
            if type(value) is int:
                constants[node.targets[0].id] = ('Integer', value)
            elif isinstance(value,tuple) and len(value)==2 and all(type(v) is int for v in value):
                constants[node.targets[0].id] = ('IntegerTuple', value)
        if isinstance(node,ast.FunctionDef) and isinstance(node.returns,ast.Name) and node.returns.id in {'Grid','Piece'}:
            types = [a.annotation.id if isinstance(a.annotation,ast.Name) else None for a in node.args.args]
            if types and all(t in {'Grid','Piece','Integer','IntegerTuple'} for t in types):
                operations.append((node.name,types))
    return sorted(operations),constants


def fit_loss(predictions, targets):
    loss = 0
    for predicted,truth in zip(predictions,targets):
        h,w = max(len(predicted),len(truth)),max(len(predicted[0]),len(truth[0]))
        mismatches = sum((predicted[i][j] if i<len(predicted) and j<len(predicted[0]) else 10) !=
                         (truth[i][j] if i<len(truth) and j<len(truth[0]) else 10)
                         for i in range(h) for j in range(w))
        loss += .5*int(predicted != truth)+.5*mismatches/(h*w)
    return loss/len(targets)


def search(source, namespace, inputs, targets, *, depth=6, width=32, attempts_per_depth=5000):
    """Trusted runtime only: arbitrary DSL expansion requires process resource caps."""
    if any(type(v) is not int or v <= 0 for v in (depth,width,attempts_per_depth)) or depth>128 or width>128 or attempts_per_depth>20000:
        raise ValueError('search limits')
    inputs,targets = tuple(map(grid,inputs)),tuple(map(grid,targets))
    if not inputs or len(inputs)!=len(targets):
        raise ValueError('paired demonstrations required')
    ops, constants = library(source)
    beam = [([],{'I':inputs})]
    best, attempted, invalid = {},0,0
    for level in range(depth):
        streams = []
        for steps,values in beam:
            for op,types in ops:
                pools = [list(values) if t in {'Grid','Piece'} else [k for k,(kind,_) in constants.items() if kind==t] for t in types]
                streams.append((steps,values,op,iter(itertools.product(*pools))))
        candidates = {}
        count = 0
        while streams and count<attempts_per_depth:
            remaining = []
            for steps,values,op,arguments in streams:
                if count>=attempts_per_depth:
                    break
                try:
                    args = next(arguments)
                except StopIteration:
                    continue
                remaining.append((steps,values,op,arguments))
                attempted += 1
                count += 1
                try:
                    predictions = tuple(grid(namespace[op](*(values[a][i] if a in values else constants[a][1] for a in args))) for i in range(len(inputs)))
                except Exception:
                    invalid += 1
                    continue
                node = f'x{len(steps)}'
                newsteps = steps+[{'id':node,'op':op,'args':list(args)}]
                score = fit_loss(predictions,targets)
                rank = (score,len(newsteps),repr(newsteps))
                if predictions not in candidates or rank<candidates[predictions][0]:
                    candidates[predictions] = (rank,newsteps,{**values,node:predictions})
                if predictions not in best or rank<best[predictions][0]:
                    best[predictions] = (rank,newsteps)
                if len(candidates)>4*width:
                    candidates = dict(sorted(candidates.items(),key=lambda item:item[1][0])[:2*width])
                if len(best)>4*max(width,4):
                    best = dict(sorted(best.items(),key=lambda item:item[1][0])[:2*max(width,4)])
            streams = remaining
        beam = [(steps,values) for _,steps,values in sorted(candidates.values(),key=lambda x:x[0])[:width]]
        if not beam:
            break
    winners = sorted(best.values(),key=lambda x:x[0])[:4]
    return {'graphs':[{'steps':steps,'output':steps[-1]['id']} for _,steps in winners],
            'training_losses':[rank[0] for rank,_ in winners], 'attempted':attempted,'invalid':invalid,
            'scope':'first_order_grid_beam_not_full_DSL', 'operations':[name for name,_ in ops]}
