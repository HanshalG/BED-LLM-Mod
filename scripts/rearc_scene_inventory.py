"""Deterministic public-grid facts, without inferred semantic roles or labels."""
from collections import Counter
from scripts.rearc_graph_worker import grid


def grid_inventory(value, component_limit=8):
    if type(component_limit) is not int or not 1<=component_limit<=8:
        raise ValueError('component limit must be 1..8')
    g=grid(value)
    h,w=len(g),len(g[0])
    counts=Counter(v for row in g for v in row)
    components={c:[] for c in counts}
    visited=set()
    for i in range(h):
        for j in range(w):
            if (i,j) in visited:
                continue
            color=g[i][j]
            stack=[(i,j)]; visited.add((i,j)); cells=[]
            while stack:
                r,c=stack.pop();cells.append((r,c))
                for a,b in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
                    if 0<=a<h and 0<=b<w and (a,b) not in visited and g[a][b]==color:
                        visited.add((a,b));stack.append((a,b))
            rs,cs=zip(*cells)
            box=[min(rs),min(cs),max(rs),max(cs)]
            components[color].append({'cells':len(cells),'bbox_inclusive':box,
                'solid_rectangle':len(cells)==(box[2]-box[0]+1)*(box[3]-box[1]+1)})
    maximum=max(counts.values())
    colors=[]
    for color in sorted(counts):
        parts=sorted(components[color],key=lambda c:(-c['cells'],c['bbox_inclusive']))
        colors.append({'color':color,'cells':counts[color],'component_count_4':len(parts),
            'singleton_components':sum(p['cells']==1 for p in parts),
            'largest_components':parts[:component_limit],
            'omitted_components':max(0,len(parts)-component_limit)})
    return {'height':h,'width':w,'coordinates':'zero-based row,column',
        'most_frequent_colors':[c for c in sorted(counts) if counts[c]==maximum],
        'most_frequent_is_background':'not_assumed','colors':colors,
        'connectivity':'four-neighbor, same color; all colors included'}
