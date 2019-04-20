import copy
import chainer as ch
from chainer import distributions as dist

class StochasticVariable:
    def sample(self):
        raise NotImplementedError()

    def condition(self, data):
        raise NotImplementedError()

    @property
    def log_prob(self):
        raise NotImplementedError()

    @property
    def dependencies(self):
        raise NotImplementedError()

    @property
    def is_sampled(self):
        raise NotImplementedError()

    @property
    def is_conditioned(self):
        raise NotImplementedError()

    @property
    def is_determined(self):
        return self.is_sampled or self.is_conditioned

class NormalStochasticVariable(StochasticVariable):
    def __init__(self, mu, sigma):
        self._mu = mu
        self._sigma = sigma
        self._is_sampled = False
        self._is_conditioned = False
        self._value = None
        self._dependencies = [dep for dep in (mu, sigma) if isinstance(dep, StochasticVariable)]

    def sample(self):
        if not self.is_sampled:
            mu = self._mu.sample() if isinstance(self._mu, StochasticVariable) else self._mu
            sigma = self._sigma.sample() if isinstance(self._sigma, StochasticVariable) else self._sigma
            d = dist.Normal(mu, sigma)
            self._value = d.sample()
            self._is_sampled = True
        return self._value

    def condition(self, data):
        assert not self.is_sampled
        assert not self.is_conditioned
        self._value = copy.copy(data)
        self._is_conditioned = True
        return self._value

    @property
    def log_prob(self):
        assert self.is_determined
        return dist.Normal(self._mu, self._sigma).log_prob(self._value)


    @property
    def is_sampled(self):
        return self._is_sampled

    @property
    def is_conditioned(self):
        return self._is_conditioned

    @property
    def dependencies(self):
        return self._dependencies
