import math
import pytest
from environments.program_induction.scalar_expression import ScalarExpression


def test_source_style_and_scalar_semantics():
    f=ScalarExpression('np.sqrt(x**2) + np.sin(np.pi/2)', ['x'])
    assert f({'x':-3})==4
    assert ScalarExpression('2**0.5', [])({})==math.sqrt(2)


@pytest.mark.parametrize('text', ["__import__('os')",'x.__class__','[x for x in y]',
                                'True', 'np.load(x)', 'lambda: x','sqrt(x=1)', 'float("nan")'])
def test_unsafe_nodes(text):
    with pytest.raises(ValueError):
        ScalarExpression(text,['x'])


@pytest.mark.parametrize('text', ['2**1000000','1/0','exp(10000)','sqrt(-1)'])
def test_numerical_failures(text):
    with pytest.raises(ValueError):
        ScalarExpression(text,[])({})


def test_resource_caps_and_inputs():
    with pytest.raises(ValueError):
        ScalarExpression('1+'*40+'1',[])
    with pytest.raises(ValueError):
        ScalarExpression('x',['x'])({'x':True})
