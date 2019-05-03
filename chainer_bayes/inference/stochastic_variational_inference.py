import chainer as ch

class StochasticVariationalInference(ch.Link):
    def __init__(self, model, guide, **params):
        super(StochasticVariationalInference, self).__init__()
        self.model = model
        self.guide = guide
        with self.init_scope():
            for name, param in params.items():
                assert name != "model"
                assert name != "guide"
                setattr(self, name, param)

    def forward(self, condition):
        return - self.surrogate_objective(condition)

    def surrogate_objective(self, condition, n_samples=10):
        return sum(self.__surrogate_objective(condition) for _ in range(n_samples)) / n_samples

    def elbo(self, condition, n_samples=10):
        return sum(self.__sample_elbo(condition) for _ in range(n_samples)) / n_samples

    def __sample_elbo(self, condition):
        log_p, log_q = self.__log_p_and_log_q(condition)
        elbo = log_p + log_q
        elbo.unchain()
        return elbo

    def __surrogate_objective(self, condition):
        log_p, log_q = self.__log_p_and_log_q(condition)
        log_p_per_q = log_p - log_q
        surrogate_objective = log_q * log_p_per_q.array
        return surrogate_objective

    def __log_p_and_log_q(self, condition):
        model_process = self.model()
        guide_process = self.guide()
        assert len(condition.keys() & guide_process.keys()) == 0
        assert len((condition.keys() | guide_process.keys()) ^ model_process.keys()) == 0
        for name, data in condition.items():
            model_process[name].condition(data)

        for name, stochastic_variable in guide_process.items():
            model_process[name].condition(stochastic_variable.sample())

        log_q = sum(stochastic_variable.log_prob_sum for stochastic_variable in guide_process.values())
        log_p = sum(stochastic_variable.log_prob_sum for stochastic_variable in model_process.values())

        return log_p, log_q
