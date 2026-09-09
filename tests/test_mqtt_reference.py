from fractions import Fraction
from itertools import product
import pytest
from scripts.mqtt_reference import Reference,evaluate


def four_worlds():
    return [('s',{('s',a):('s',str(y)) for a,y in enumerate(row)})
            for row in ((0,0,0),(0,1,0),(1,0,0),(1,0,1))]


def independent_sequence_risk(machines,seq,targets):
    histories={}
    for i,(start,table) in enumerate(machines):
        state=start;out=[]
        for a in seq:
            if a==3:state=start;y='RESET_ACK'
            else:state,y=table[state,a]
            out.append(y)
        histories.setdefault(tuple(out),[]).append(i)
    # This fixture's target trace uniquely identifies each machine.
    return sum((Fraction(len(ids),4)*Fraction(len(ids)-1,2*len(ids))
                for ids in histories.values()),Fraction())


def test_adaptive_and_committed_against_independent_enumeration():
    ms=four_worlds();r=Reference(ms,('a','b','c'),[(0,1,2)])
    try:
        assert r.risk((0,1,2,3))==Fraction(3,8)
        assert r.optimal(r.initial,2)[0]==0
        expected=min((independent_sequence_risk(ms,s,None),s) for s in product(range(4),repeat=2))
        assert r.committed((r.initial,),2)==expected
        assert expected[0]==Fraction(1,8)
        assert r.deployed(r.initial,2,2)==0
    finally:r.clear()


def test_physical_state_and_reset_preserve_posterior():
    ms=[('s',{('s',0):('t','ack'),('s',1):('s','idle'),
              ('t',0):('t','ack'),('t',1):('t',str(i))}) for i in range(2)]
    r=Reference(ms,('arm','probe'),[(0,1)])
    try:
        first=r.branches(r.initial,0)
        assert len(first)==1
        learned=r.branches(first[0],1)
        assert len(learned)==2
        assert r.branches(learned[0],r.reset)==(((0,'s'),),)
        assert len(r.branches(r.initial,1))==1
        assert r.optimal(r.initial,1)[0]==Fraction(1,4)
        assert r.optimal(r.initial,2)[0]==0
    finally:r.clear()


def test_singleton_is_zero_not_undefined():
    out=evaluate(four_worlds()[:1],('a','b','c'),[(0,1,2)],budget=2)
    assert out['initial_risk']=='0' and set(out['risk'].values())=={'0'}


def test_cap_stops_before_response_calculation():
    with pytest.raises(TimeoutError):Reference(four_worlds(),('a','b','c'),[(0,)],seconds=-1)


def test_stateful_values_against_uncached_explicit_world_tree():
    import random
    from collections import Counter
    rng=random.Random(771)
    for _ in range(5):
        ms=[(0,{(s,a):(rng.randrange(2),str(rng.randrange(2)))
                 for s in range(2) for a in range(2)}) for i in range(4)]
        targets=list(product(range(2),repeat=3))
        def trace(i,seq):
            s=0;ys=[]
            for a in seq:
                if a==2:s=0;y='RESET_ACK'
                else:s,y=ms[i][1][s,a]
                ys.append(y)
            return tuple(ys)
        def loss(ids):
            return sum((Fraction(1,2)-sum((Fraction(n,len(ids))**2/2
                for n in Counter(trace(i,t) for i in ids).values()),Fraction())
                for t in targets),Fraction())/len(targets)
        def children(ids,seq,a):
            groups={}
            for i in ids:groups.setdefault(trace(i,seq+(a,)),[]).append(i)
            return list(groups.values())
        def opt(ids,seq,depth):
            if depth==0:return loss(ids),None
            return min((sum((Fraction(len(b),len(ids))*opt(b,seq+(a,),depth-1)[0]
                      for b in children(ids,seq,a)),Fraction()),a) for a in range(3))
        def policy(ids,seq,left,h):
            if not left:return loss(ids)
            a=opt(ids,seq,min(left,h))[1]
            return sum((Fraction(len(b),len(ids))*policy(b,seq+(a,),left-1,h)
                        for b in children(ids,seq,a)),Fraction())
        r=Reference(ms,('a','b'),targets)
        try:
            ids=list(range(4))
            assert r.optimal(r.initial,3)==opt(ids,(),3)
            for h in (1,2,3):assert r.deployed(r.initial,3,h)==policy(ids,(),3,h)
            fixed=[]
            for seq in product(range(3),repeat=3):
                groups={}
                for i in ids:groups.setdefault(trace(i,seq),[]).append(i)
                fixed.append((sum((Fraction(len(b),4)*loss(b) for b in groups.values()),Fraction()),seq))
            assert r.committed((r.initial,),3)==min(fixed)
            assert r.random(r.initial,3)==sum((v for v,s in fixed),Fraction())/27
        finally:r.clear()
