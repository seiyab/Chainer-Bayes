import chainer as ch
from chainer import distributions as dist

from stochastic_variable import StochasticVariable, NormalStochasticVariable

class Distribution:
    def __init__(self, distribution_class, *params):
        self._distribution_class = distribution_class
        self._params = params

    def __call__(self):
        return StochasticVariable(self._distribution_class, *self._params)

class Normal(Distribution):
    def __init__(self, mu, sigma):
        Distribution.__init__(self, dist.Normal, mu, sigma)
