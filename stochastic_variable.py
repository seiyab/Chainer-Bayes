import copy
import chainer as ch
from chainer import distributions as dist

class StochasticVariable(object):
    def __init__(self, distribution_class, *params):
        self._distribution_class = distribution_class
        self._params = params

        self._is_sampled = False
        self._is_conditioned = False
        self._value = None
        self._dependencies = [param for param in self._params if isinstance(param, StochasticVariable)]

    def sample(self):
        if not self.is_determined:
            realized_params = [
                param.sample() if isinstance(param, StochasticVariable) else param
                for param in self._params
            ]
            distribution = self._distribution_class(*realized_params)
            self._value = distribution.sample()
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
        assert all(dependency.is_determined for dependency in self._dependencies)
        realized_params = [
            param._value if isinstance(param, StochasticVariable) else param
            for param in self._params
        ]
        return self._distribution_class(*realized_params).log_prob(self._value)

    @property
    def is_sampled(self):
        return self._is_sampled

    @property
    def is_conditioned(self):
        return self._is_conditioned

    @property
    def dependencies(self):
        return self._dependencies

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
