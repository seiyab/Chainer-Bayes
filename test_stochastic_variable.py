import numpy as np
import chainer as ch
from chainer import distributions as chdist
from distribution import Normal, Distribution
from stochastic_variable import StochasticVariable

def run():
    test_sample()
    test_sample_size()
    test_dependency()
    test_condition()
    test_log_prob()
    test_sample_bernoulli()

def test_sample():
    normal_dist = Normal(ch.Variable(np.zeros(10)), ch.Variable(np.ones(10)))
    normal_sv = normal_dist()

    assert not normal_sv.is_determined

    sample = normal_sv.sample()

    assert normal_sv.is_sampled
    assert normal_sv.is_determined
    assert isinstance(sample, ch.Variable)
    assert sample.shape == (10,)
    assert all(normal_sv.sample().data == normal_sv.sample().data)
    assert any(normal_dist().sample().data != normal_dist().sample().data)

def test_sample_size():
    a = ch.Variable(np.ones(3))
    b = ch.Variable(np.ones(3))
    beta = StochasticVariable(chdist.Beta, a, b, sample_shape=(5, 4))

    x = beta.sample()
    assert x.shape == (5, 4, 3)

def test_sample_bernoulli():
    bernoulli_dist = Distribution(ch.distributions.Bernoulli, ch.Variable(np.ones(5) * 0.5))
    bernoulli_sv = bernoulli_dist()

    assert not bernoulli_sv.is_determined

    sample = bernoulli_sv.sample()

    assert bernoulli_sv.is_determined
    assert isinstance(sample, ch.Variable)
    assert sample.shape == (5,)
    assert all(bernoulli_sv.sample().data == bernoulli_sv.sample().data)

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
