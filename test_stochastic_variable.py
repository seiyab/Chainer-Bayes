import numpy as np
import chainer as ch
from chainer import distributions as dist
from stochastic_variable import StochasticVariable

def run():
    test_sample()
    test_sample_size()
    test_dependency()
    test_condition()
    test_log_prob()
    test_sample_bernoulli()

def test_sample():
    mu = ch.Variable(np.zeros(10))
    sigma = ch.Variable(np.ones(10))
    x = StochasticVariable(dist.Normal, mu, sigma)

    assert not x.is_determined

    sample = x.sample()

    assert x.is_sampled
    assert x.is_determined
    assert isinstance(sample, ch.Variable)
    assert sample.shape == (10,)
    assert all(x.sample().data == x.sample().data)

def test_sample_size():
    a = ch.Variable(np.ones(3))
    b = ch.Variable(np.ones(3))
    beta = StochasticVariable(dist.Beta, a, b, sample_shape=(5, 4))

    x = beta.sample()
    assert x.shape == (5, 4, 3)

def test_sample_bernoulli():
    p = StochasticVariable(dist.Bernoulli, ch.Variable(np.ones(5) * 0.5))

    assert not p.is_determined

    sample = p.sample()

    assert p.is_determined
    assert isinstance(sample, ch.Variable)
    assert sample.shape == (5,)
    assert all(p.sample().data == p.sample().data)

def test_dependency():
    x_mu = ch.Variable(np.zeros(10))
    x_sigma = ch.Variable(np.ones(10))
    x = StochasticVariable(dist.Normal, x_mu, x_sigma)

    y_sigma = ch.Variable(np.ones(10))
    y = StochasticVariable(dist.Normal, x, y_sigma)
    assert x in y.dependencies
    sample = y.sample()
    assert x.is_sampled
    assert y.is_sampled

def test_condition():
    mu = ch.Variable(np.zeros(10))
    sigma = ch.Variable(np.ones(10))
    x = StochasticVariable(dist.Normal, mu, sigma)
    assert not x.is_determined
    sample = x.condition(ch.Variable(np.ones(10)))
    assert all(sample.data == np.ones(10))
    assert x.is_conditioned
    assert x.is_determined

def test_log_prob():
    mu = ch.Variable(np.array(0.))
    sigma = ch.Variable(np.array(1.))
    cond_value = ch.Variable(np.array(0.5))

    x = StochasticVariable(dist.Normal, mu, sigma)
    x.condition(cond_value)
    assert x.log_prob.data == dist.Normal(mu, sigma).log_prob(cond_value).data
