import copy
import chainer as ch
from chainer import distributions as dist

class StochasticVariable(object):
    def __init__(self, distribution_class, *params, **kwparams):
        self._distribution_class = distribution_class
        self._params = params
        self._sample_shape = kwparams.pop('sample_shape', ())
        self._kwparams = kwparams

        self._is_sampled = False
        self._is_conditioned = False
        self._value = None
        self._dependencies = [param for param in self._params if isinstance(param, StochasticVariable)] \
                + [param for param in self._kwparams.values() if isinstance(param, StochasticVariable)]

    def sample(self):
        if not self.is_determined:
            realized_params = [
                    param.sample() if isinstance(param, StochasticVariable) else param
                    for param in self._params
                    ]
            realized_kwparams = {
                    key: param.sample if isinstance(param, StochasticVariable) else param
                    for key, param in self._kwparams.items()
                    }
            distribution = self._distribution_class(*realized_params, **realized_kwparams)
            self._value = distribution.sample(sample_shape=self._sample_shape)
            self._is_sampled = True
            self._value.unchain()
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
        realized_kwparams = {
                key: param._value if isinstance(param, StochasticVariable) else param
                for key, param in self._kwparams.items()
        }
        return self._distribution_class(*realized_params, **realized_kwparams).log_prob(self._value)

    @property
    def log_prob_sum(self):
        return self._value.xp.sum(self.log_prob)

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
