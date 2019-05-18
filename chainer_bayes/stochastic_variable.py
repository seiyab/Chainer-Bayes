import copy
import chainer as ch
from chainer import distributions as dist

from chainer_bayes import lazy

class StochasticVariable(lazy.LazyNode):
    def __init__(self, distribution_class, *params, **kwparams):
        self.__distribution_class = distribution_class
        self.__params = params
        self.__sample_shape = kwparams.pop('sample_shape', ())
        self.__kwparams = kwparams

        self.__is_sampled = False
        self.__is_conditioned = False
        self.__value = None

    def evaluate(self):
        return self.sample()

    @property
    def is_evaluated(self):
        return self.is_sampled or self.is_conditioned

    def sample(self):
        if not self.is_evaluated:
            realized_params = [lazy.realize(param) for param in self.__params]
            realized_kwparams = {key: lazy.realize(param) for key, param in self.__kwparams.items()}
            distribution = self.__distribution_class(*realized_params, **realized_kwparams)
            self.__value = distribution.sample(sample_shape=self.__sample_shape)
            self.__is_sampled = True
            self.__value.unchain()
        return self.__value

    def condition(self, data):
        assert not self.is_sampled
        assert not self.is_conditioned
        self.__value = copy.copy(data)
        self.__is_conditioned = True
        return self.__value

    @property
    def log_prob(self):
        assert self.is_evaluated
        assert all(param.is_evaluated for param in self.dependencies if issubclass(type(param), lazy.LazyNode))
        realized_params = [lazy.realize(param) for param in self.__params]
        realized_kwparams = {key: lazy.realize(param) for key, param in self.__kwparams.items()}
        return self.__distribution_class(*realized_params, **realized_kwparams).log_prob(self.__value)

    @property
    def log_prob_sum(self):
        return self.__value.xp.sum(self.log_prob)

    @property
    def is_sampled(self):
        return self.__is_sampled

    @property
    def is_conditioned(self):
        return self.__is_conditioned

    @property
    def dependencies(self):
        return [*self.__params, *self.__kwparams.values()]