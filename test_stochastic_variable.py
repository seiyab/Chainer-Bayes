import numpy as np
import chainer as ch
from chainer import distributions as chdist
from distribution import Normal

def run():
    test_sample()
    test_dependency()
    test_condition()
    test_log_prob()


def test_sample():
    normal_dist = Normal(ch.Variable(np.zeros(10)), ch.Variable(np.ones(10)))
    normal_sv = normal_dist()
    sample = normal_sv.sample()

    assert normal_sv.is_sampled
    assert normal_sv.is_determined
    assert isinstance(sample, ch.Variable)
    assert sample.shape == (10,)
    assert all(normal_sv.sample().data == normal_sv.sample().data)
    assert any(normal_dist().sample().data != normal_dist().sample().data)


def test_dependency():
    normal_dist = Normal(ch.Variable(np.zeros(10)), ch.Variable(np.ones(10)))
    normal_sv = normal_dist()
    sv_depends_on_sv = Normal(normal_sv, ch.Variable(np.ones(10)))()
    assert normal_sv in sv_depends_on_sv.dependencies
    sample = sv_depends_on_sv.sample()
    assert normal_sv.is_sampled

def test_condition():
    normal_dist = Normal(ch.Variable(np.zeros(10)), ch.Variable(np.ones(10)))
    normal_sv = normal_dist()
    assert not normal_sv.is_determined
    sample = normal_sv.condition(ch.Variable(np.ones(10)))
    assert all(sample.data == np.ones(10))
    assert normal_sv.is_conditioned
    assert normal_sv.is_determined

def test_log_prob():
    mean = ch.Variable(np.array(0.))
    std = ch.Variable(np.array(1.))
    cond_value = ch.Variable(np.array(0.5))

    normal_dist = Normal(mean, std)
    normal_sv = normal_dist()
    normal_sv.condition(cond_value)
    assert normal_sv.log_prob.data == chdist.Normal(mean, std).log_prob(cond_value).data

test()
