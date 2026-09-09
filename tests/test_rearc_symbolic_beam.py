from scripts.rearc_symbolic_beam import search,library

SOURCE = '''ZERO=0
ONE=1
def vmirror(x: Grid) -> Grid: pass
def hconcat(a: Grid,b: Grid) -> Grid: pass
def objects(x: Grid) -> Objects: pass
'''
NAMESPACE = {'vmirror':lambda g:tuple(tuple(reversed(r)) for r in g),
             'hconcat':lambda a,b:tuple(x+y for x,y in zip(a,b))}


def test_search_finds_composition_from_demonstrations():
    result=search(SOURCE,NAMESPACE,[[[1,2]],[[3,4]]],[[[1,2,2,1]],[[3,4,4,3]]],depth=2,width=16,attempts_per_depth=100)
    assert result['training_losses'][0]==0
    assert len(result['graphs'][0]['steps'])==2
    assert result['attempted']<=200


def test_no_reference_or_target_bank_argument_and_library_limits_explicit():
    operations,_=library(SOURCE)
    assert {name for name,_ in operations}=={'vmirror','hconcat'}
    a=search(SOURCE,NAMESPACE,[[[1,2]]],[[[2,1]]],depth=1,width=2,attempts_per_depth=10)
    b=search(SOURCE,NAMESPACE,[[[1,2]]],[[[2,1]]],depth=1,width=2,attempts_per_depth=10)
    assert a==b
    assert a['training_losses'][0]==0
