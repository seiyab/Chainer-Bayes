import unittest

import numpy as np
import chainer as ch
from chainer import distributions as dist
from chainer_bayes import StochasticVariable

class TestStochasticVariable(unittest.TestCase):

    def test_sample(self):
        mu = ch.Variable(np.zeros(10))
        sigma = ch.Variable(np.ones(10))
        x = StochasticVariable(dist.Normal, mu, sigma)

        self.assertFalse(x.is_evaluated)

        sample = x.sample()

        self.assertTrue(x.is_sampled)
        self.assertTrue(x.is_evaluated)
        self.assertIsInstance(sample, ch.Variable)
        self.assertEqual(sample.shape, (10,))
        self.assertTrue(all(x.sample().data == x.sample().data))

    def test_sample_size(self):
        a = ch.Variable(np.ones(3))
        b = ch.Variable(np.ones(3))
        beta = StochasticVariable(dist.Beta, a, b, sample_shape=(5, 4))

        x = beta.sample()
        self.assertEqual(x.shape, (5, 4, 3))

    def test_sample_bernoulli(self):
        p = StochasticVariable(dist.Bernoulli, ch.Variable(np.ones(5) * 0.5))

        self.assertFalse(p.is_evaluated)

        sample = p.sample()

        self.assertTrue(p.is_evaluated)
        self.assertIsInstance(sample, ch.Variable)
        self.assertEqual(sample.shape, (5,))
        self.assertTrue(all(p.sample().data == p.sample().data))

    def test_dependency(self):
        x_mu = ch.Variable(np.zeros(10))
        x_sigma = ch.Variable(np.ones(10))
        x = StochasticVariable(dist.Normal, x_mu, x_sigma)

        y_sigma = ch.Variable(np.ones(10))
        y = StochasticVariable(dist.Normal, x, y_sigma)

        sample = y.sample()

        self.assertTrue(x.is_sampled)
        self.assertTrue(y.is_sampled)

    def test_condition(self):
        mu = ch.Variable(np.zeros(10))
        sigma = ch.Variable(np.ones(10))
        x = StochasticVariable(dist.Normal, mu, sigma)

        self.assertFalse(x.is_evaluated)

        sample = x.condition(ch.Variable(np.ones(10)))

        self.assertTrue(all(sample.data == np.ones(10)))
        self.assertTrue(x.is_conditioned)
        self.assertTrue(x.is_evaluated)

    def test_log_prob(self):
        mu = ch.Variable(np.array(0.))
        sigma = ch.Variable(np.array(1.))
        cond_value = ch.Variable(np.array(0.5))

        x = StochasticVariable(dist.Normal, mu, sigma)
        x.condition(cond_value)
        self.assertEqual(x.log_prob.data, dist.Normal(mu, sigma).log_prob(cond_value).data)
