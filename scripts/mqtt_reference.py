"""Exact finite-prior stateful reference; full words do not get free steps."""
from collections import Counter
from fractions import Fraction
from functools import lru_cache
from time import monotonic


class Reference:
    def __init__(self, machines, alphabet, target_words, seconds=180):
        self.machines=machines
        self.alphabet=tuple(alphabet)
        if not machines or not alphabet or not target_words:
            raise ValueError('nonempty reference')
        self.actions=tuple(range(len(alphabet)+1))
        self.reset=len(alphabet)
        self.initial=tuple((i,m[0]) for i,m in enumerate(machines))
        self.deadline=monotonic()+seconds
        self.targets=[]
        for start,table in machines:
            row=[]
            for word in target_words:
                state=start;out=[]
                for action in word:
                    self.check()
                    state,y=table[state,action];out.append(y)
                row.append(tuple(out))
            self.targets.append(tuple(row))
        self.supports=set()

    def check(self):
        if monotonic()>self.deadline: raise TimeoutError('whole-group reference cap')

    @lru_cache(maxsize=100000)
    def risk(self, ids):
        self.check()
        if not ids: raise ValueError('empty posterior')
        n=len(ids);total=0
        for j in range(len(self.targets[0])):
            counts=Counter(self.targets[i][j] for i in ids)
            total+=n*n-sum(c*c for c in counts.values())
        return Fraction(total,2*n*n*len(self.targets[0]))

    @lru_cache(maxsize=100000)
    def branches(self, state, action):
        self.check()
        if action not in self.actions or not state: raise ValueError('state/action')
        self.supports.add(tuple(i for i,s in state))
        groups={}
        for i,s in state:
            if action==self.reset: t,y=self.machines[i][0],'RESET_ACK'
            else: t,y=self.machines[i][1][s,action]
            groups.setdefault(y,[]).append((i,t))
        return tuple(tuple(v) for k,v in sorted(groups.items()))

    @lru_cache(maxsize=100000)
    def optimal(self,state,depth):
        self.check()
        if not depth:return self.risk(tuple(i for i,s in state)),None
        return min((sum((Fraction(len(b),len(state))*self.optimal(b,depth-1)[0]
                         for b in self.branches(state,a)),Fraction()),a) for a in self.actions)

    @lru_cache(maxsize=100000)
    def deployed(self,state,budget,horizon):
        self.check()
        if not budget:return self.risk(tuple(i for i,s in state))
        a=self.optimal(state,min(budget,horizon))[1]
        return sum((Fraction(len(b),len(state))*self.deployed(b,budget-1,horizon)
                    for b in self.branches(state,a)),Fraction())

    @lru_cache(maxsize=100000)
    def random(self,state,budget):
        self.check()
        if not budget:return self.risk(tuple(i for i,s in state))
        return sum((Fraction(len(b),len(state)*len(self.actions))*self.random(b,budget-1)
                    for a in self.actions for b in self.branches(state,a)),Fraction())

    @lru_cache(maxsize=100000)
    def committed(self,forest,depth):
        self.check()
        n=sum(len(s) for s in forest)
        if not depth:
            return sum((Fraction(len(s),n)*self.risk(tuple(i for i,t in s))
                        for s in forest),Fraction()),()
        best=None
        for a in self.actions:
            # All possible histories must take the SAME next action. A forest
            # retains their distinct posterior supports and physical states.
            next_forest=tuple(sorted(b for s in forest for b in self.branches(s,a)))
            value,tail=self.committed(next_forest,depth-1)
            candidate=value,(a,)+tail
            if best is None or candidate<best:best=candidate
        return best

    def clear(self):
        for name in ('risk','branches','optimal','deployed','random','committed'):
            getattr(self,name).cache_clear()


def evaluate(machines,alphabet,targets,budget=6,seconds=180):
    start=monotonic()
    ref=Reference(machines,alphabet,targets,seconds)
    try:
        initial=ref.risk(tuple(range(len(machines))))
        values={f'h{h}':ref.deployed(ref.initial,budget,h) for h in (1,2,3)}
        values['adaptive_full']=ref.optimal(ref.initial,budget)[0]
        values['committed'],sequence=ref.committed((ref.initial,),budget)
        values['random']=ref.random(ref.initial,budget)
        if any(values['adaptive_full']>v for v in values.values()):
            raise AssertionError('adaptive optimum violated')
        if any(v>initial for v in values.values()):
            raise AssertionError('information increased Bayes risk')
        return {'status':'complete','initial_risk':str(initial),
                'risk':{k:str(v) for k,v in values.items()},
                'root_actions':{f'h{h}':ref.optimal(ref.initial,min(h,budget))[1] for h in (1,2,3)},
                'committed_sequence':sequence,'posterior_supports':len(ref.supports),
                'posterior_sizes':sorted({len(s) for s in ref.supports}),
                'work':{name:getattr(ref,name).cache_info()._asdict() for name in
                        ('risk','branches','optimal','deployed','random','committed')},
                'elapsed_seconds':monotonic()-start}
    finally:ref.clear()
