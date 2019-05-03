import unittest

from chainer_bayes.inference.maximum_likelihood_estimation import MaximumLikelihoodEstimation
from chainer_bayes.stochastic_variable import StochasticVariable

import numpy as np
import chainer as ch
from chainer import distributions as dist

class TestMaximumLikelihoodEstimation(unittest.TestCase):

    def test_maximum_likelihood_one_step(self):
        np.random.seed(0)

        param_mu_init = 3.
        param_sigma_init = 2.
        param_mu = ch.Parameter(np.array(param_mu_init))
        param_sigma = ch.Parameter(np.array(param_sigma_init))

        model = lambda: {"x": StochasticVariable(dist.Normal, param_mu, param_sigma, sample_shape=(30,))}
        infer = MaximumLikelihoodEstimation(model, mu=param_mu)
        optimizer = ch.optimizers.SGD(lr=0.05).setup(infer)
    
        data = ch.Variable(np.random.randn(30))
        infer.cleargrads()
        loss_before = infer({"x": data})
        loss_before.backward()
        optimizer.update()

        loss_after = infer({"x": data})

        self.assertLess(param_mu.data, param_mu_init)
        self.assertEqual(param_sigma.data, param_sigma_init)
        self.assertGreater(loss_before.data, loss_after.data)

    def test_maximum_likelihood_fixed_point(self):
        np.random.seed(0)

        param_mu_init = 0.
        param_sigma_init = 1.
        param_mu = ch.Parameter(np.array(param_mu_init))
        param_sigma = ch.Parameter(np.array(param_sigma_init))

        model = lambda: {"x": StochasticVariable(dist.Normal, param_mu, param_sigma, sample_shape=(30,))}
        infer = MaximumLikelihoodEstimation(model, mu=param_mu, sigma=param_sigma)
        optimizer = ch.optimizers.SGD(lr=0.05).setup(infer)

        for _ in range(10):
            data = ch.Variable(np.random.randn(30))
            infer.cleargrads()
            loss = infer({"x": data})
            loss.backward()
            optimizer.update()

        self.assertAlmostEqual(param_mu.data, 0, delta=0.3)
        self.assertAlmostEqual(param_sigma.data, 1, delta=0.2)

    def test_maxlimum_likelihood_categorical(self):
        np.random.seed(0)

        param_logit_init = np.zeros(3)
        param_logit = ch.Parameter(np.array(param_logit_init))

        model = lambda: {"x": StochasticVariable(dist.Categorical, logit=param_logit, sample_shape=(10,))}
        infer = MaximumLikelihoodEstimation(model, logit=param_logit)
        optimizer = ch.optimizers.SGD(lr=0.05).setup(infer)

        data = ch.Variable(np.array([0.]*5 + [1.]*5))
        infer.cleargrads()
        loss = infer({"x": data})
        loss.backward()
        optimizer.update()

        self.assertLess(param_logit_init[0], param_logit[0].data)
