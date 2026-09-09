"""Complete one/two-operation string grammar using standard string operations."""
from itertools import product
import re


def operations():
    ops = [(name, getattr(str, name)) for name in
           ('lower', 'upper', 'title', 'swapcase', 'strip')]
    ops += [('identity', lambda x:x), ('reverse', lambda x:x[::-1])]
    for name, pattern in [('letters',r'[^A-Za-z]'), ('digits',r'[^0-9]'),
                          ('upper_letters',r'[^A-Z]'), ('lower_letters',r'[^a-z]'),
                          ('remove_space',r'\s')]:
        ops.append((name, lambda x,p=pattern:re.sub(p,'',x)))
    for k in range(1,7):
        ops.extend([(f'prefix_{k}', lambda x,k=k:x[:k]),
                    (f'drop_first_{k}', lambda x,k=k:x[k:]),
                    (f'suffix_{k}', lambda x,k=k:x[-k:]),
                    (f'drop_last_{k}', lambda x,k=k:x[:-k])])
    for sep in (' ', ',', '-', '_', '.', '/', '@'):
        for idx in (0,1,-1):
            def select(x,s=sep,i=idx):
                parts=x.split(s)
                return parts[i] if -len(parts)<=i<len(parts) else ''
            ops.append((f'split_{sep!r}_{idx}',select))
    ops.extend([('first_word',lambda x:x.split()[0] if x.split() else ''),
                ('last_word',lambda x:x.split()[-1] if x.split() else ''),
                ('initials',lambda x:''.join(w[0] for w in x.split()))])
    return tuple(ops)


def support(initial, inputs):
    """Uniform length, uniform syntax within length; exact integer multiplicities.

    Only one observed output enters. Query and target outputs are not accepted.
    One-operation syntax has N times the mass of each two-operation syntax.
    """
    if (type(initial) is not tuple or len(initial)!=2 or
            any(type(x) is not str for x in initial) or
            type(inputs) is not list or any(type(x) is not str for x in inputs)):
        raise ValueError('one string pair and string-only evaluation inputs required')
    ops=operations()
    rows,programs=[],[]
    attempted=0
    for length in (1,2):
        for indices in product(range(len(ops)),repeat=length):
            attempted+=1
            def evaluate(x):
                for i in indices:
                    x=ops[i][1](x)
                return x
            if evaluate(initial[0])!=initial[1]:
                continue
            multiplicity=len(ops) if length==1 else 1
            row=[evaluate(x) for x in inputs]
            rows.extend([row]*multiplicity)
            programs.append(dict(operations=[ops[i][0] for i in indices],multiplicity=multiplicity))
    return rows,dict(operations=len(ops),attempted=attempted,compatible_syntax=len(programs),
                    prior_units=len(rows),unconditioned_prior_units=2*len(ops)**2,
                    programs=programs,complete_declared_grammar=True,
                    complete_human_task_prior=False)
