import numpy as np
import chainer as ch
from chainer import distributions as chdist
from distribution import Normal, Distribution
from inference import MaximumLilekihoodEstimation

def run():
    test_maximum_likelihood_small_steps()
    test_maximum_likelihood_fixed_point()
    test_maxlimum_likelihood_bernoulli()

def test_maximum_likelihood_small_steps():
    np.random.seed(0)

    param_mu_init = 3.
    param_sigma_init = 2.
    param_mu = ch.Parameter(np.array(param_mu_init))
    param_sigma = ch.Parameter(np.array(param_sigma_init))

    model = lambda: {"x": Normal(param_mu, param_sigma)()}
    infer = MaximumLilekihoodEstimation(model, mu=param_mu, sigma=param_sigma)
    optimizer = ch.optimizers.SGD(lr=0.05).setup(infer)

    data = ch.Variable(np.random.randn(30))
    for d in data:
        infer.cleargrads()
        loss = infer({"x": d})
        loss.backward()
        optimizer.update()

    eps = 0.2
    assert 0-eps < param_mu.data < param_mu_init

def test_maximum_likelihood_fixed_point():
    np.random.seed(0)

    param_mu_init = 0.
    param_sigma_init = 1.
    param_mu = ch.Parameter(np.array(param_mu_init))
    param_sigma = ch.Parameter(np.array(param_sigma_init))

    model = lambda: {"x": Normal(param_mu, param_sigma)()}
    infer = MaximumLilekihoodEstimation(model, mu=param_mu, sigma=param_sigma)
    optimizer = ch.optimizers.SGD(lr=0.05).setup(infer)

    data = ch.Variable(np.random.randn(50))
    for d in data:
        infer.cleargrads()
        loss = infer({"x": d})
        loss.backward()
        optimizer.update()

    eps = 0.3
    assert 0-eps < param_mu.data < 0+eps
    assert 1-eps < param_sigma.data < 1+eps

def test_maxlimum_likelihood_bernoulli():
    np.random.seed(0)

    param_p_init = 0.3
    param_p = ch.Parameter(np.array(param_p_init))

    model = lambda: {"b": Distribution(ch.distributions.Bernoulli, param_p)()}
    infer = MaximumLilekihoodEstimation(model, p=param_p)
    optimizer = ch.optimizers.SGD(lr=0.05).setup(infer)

    data = ch.Variable(np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float))
    for d in data:
        infer.cleargrads()
        loss = infer({"b": d})
        loss.backward()
        optimizer.update()

    eps = 0.1

    assert param_p_init < param_p.data < 0.5+eps

