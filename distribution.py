import chainer as ch
from chainer import distributions as dist

from stochastic_variable import StochasticVariable, NormalStochasticVariable

class Distribution:
    pass

class Normal(Distribution):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
    def __call__(self):
        return NormalStochasticVariable(self.mu, self.sigma)
