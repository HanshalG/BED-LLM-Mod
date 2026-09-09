from fractions import Fraction
from scripts.mqtt_committed_certificate import certificate
from tests.test_mqtt_reference import four_worlds


def test_certificate_retains_unresolved_predictive_mass():
    machines=four_worlds()
    loss,sizes=certificate(machines,[(0,1,2)],(0,1),3)
    assert loss==Fraction(1,8)
    assert sizes==[1,1,2]
    assert certificate(machines,[(0,1,2)],(0,1,2),3)[0]==0


def test_distinct_models_can_have_zero_predictive_risk():
    machines=four_worlds()[:2]
    loss,sizes=certificate(machines,[(0,)],(3,),3)
    assert loss==0 and sizes==[2]
