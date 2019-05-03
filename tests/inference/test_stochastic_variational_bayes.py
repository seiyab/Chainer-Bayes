import unittest

from chainer_bayes.inference.stochastic_variational_inference import StochasticVariationalInference
from chainer_bayes.stochastic_variable import StochasticVariable

import numpy as np
import chainer as ch
from chainer import distributions as dist

class TestStochasticVariationalInference(unittest.TestCase):
    def test_svi(self):
        np.random.seed(0)
        n_obs = 30
        n_samples = 30
        actual_sigma = np.array(2.)
        actual_mu = np.array(0.5)

        # generative model
        def model():
            m = ch.Variable(np.array(0.))
            s = ch.Variable(np.array(3.))
            mu = StochasticVariable(dist.Normal, m, s)
            sigma = ch.Variable(actual_sigma)

            return {
                    "mu": mu,
                    "x": StochasticVariable(dist.Normal, mu, sigma, sample_shape=(n_obs,))
                    }

        # variational posterior
        param_mu = ch.Parameter(np.array(0.))
        param_sigma = ch.Parameter(np.array(3.))
        def guide():
            return {"mu": StochasticVariable(dist.Normal, param_mu, param_sigma)}

        # setup inference
        infer = StochasticVariationalInference(model, guide, param_mu=param_mu, param_sigma=param_sigma)
        optimizer = ch.optimizers.SGD(lr=0.01).setup(infer)

        # setup data
        data = ch.Variable(np.random.randn(n_obs)*actual_sigma + actual_mu)
        condition = {"x": data}

        before_elbo = infer.elbo(condition, n_samples=n_samples)

        # update single step
        infer.cleargrads()
        loss = -infer.surrogate_objective(condition, n_samples=n_samples)
        loss.backward()
        optimizer.update()

        after_elbo = infer.elbo(condition, n_samples=n_samples)

        self.assertGreater(after_elbo.data, before_elbo.data)
