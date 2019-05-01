import numpy as np
import chainer as ch
from chainer import distributions as dist
from stochastic_variable import StochasticVariable
from inference import MaximumLilekihoodEstimation

def run():
    test_maximum_likelihood_one_step()
    test_maximum_likelihood_fixed_point()
    test_maxlimum_likelihood_categorical()

def test_maximum_likelihood_one_step():
    np.random.seed(0)

    param_mu_init = 3.
    param_sigma_init = 2.
    param_mu = ch.Parameter(np.array(param_mu_init))
    param_sigma = ch.Parameter(np.array(param_sigma_init))

    model = lambda: {"x": StochasticVariable(dist.Normal, param_mu, param_sigma, sample_shape=(30,))}
    infer = MaximumLilekihoodEstimation(model, mu=param_mu)
    optimizer = ch.optimizers.SGD(lr=0.05).setup(infer)

    data = ch.Variable(np.random.randn(30))
    infer.cleargrads()
    loss_before = infer({"x": data})
    loss_before.backward()
    optimizer.update()

    loss_after = infer({"x": data})

    assert param_mu.data < param_mu_init
    assert param_sigma.data == param_sigma_init
    assert loss_before.data > loss_after.data

def test_maximum_likelihood_fixed_point():
    np.random.seed(0)

    param_mu_init = 0.
    param_sigma_init = 1.
    param_mu = ch.Parameter(np.array(param_mu_init))
    param_sigma = ch.Parameter(np.array(param_sigma_init))

    model = lambda: {"x": StochasticVariable(dist.Normal, param_mu, param_sigma, sample_shape=(30,))}
    infer = MaximumLilekihoodEstimation(model, mu=param_mu, sigma=param_sigma)
    optimizer = ch.optimizers.SGD(lr=0.05).setup(infer)

    for _ in range(10):
        data = ch.Variable(np.random.randn(30))
        infer.cleargrads()
        loss = infer({"x": data})
        loss.backward()
        optimizer.update()

    eps = 0.3
    assert 0-eps < param_mu.data < 0+eps
    assert 1-eps < param_sigma.data < 1+eps

def test_maxlimum_likelihood_categorical():
    np.random.seed(0)

    param_logit_init = np.zeros(3)
    param_logit = ch.Parameter(np.array(param_logit_init))

    model = lambda: {"x": StochasticVariable(dist.Categorical, logit=param_logit, sample_shape=(10,))}
    infer = MaximumLilekihoodEstimation(model, logit=param_logit)
    optimizer = ch.optimizers.SGD(lr=0.05).setup(infer)

    data = ch.Variable(np.array([0.]*5 + [1.]*5))
    infer.cleargrads()
    loss = infer({"x": data})
    loss.backward()
    optimizer.update()

    assert param_logit_init[0] < param_logit[0].data

