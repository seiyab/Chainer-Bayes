import chainer as ch

class MaximumLilekihoodEstimation(ch.Link):
    def __init__(self, model, **params):
        super(MaximumLilekihoodEstimation, self).__init__()
        self.model = model
        with self.init_scope():
            for name, param in params.items():
                assert name != "model"
                setattr(self, name, param)

    def forward(self, condition):
        process = self.model()
        for name, data in condition.items():
            process[name].condition(data)
        for sv in process.values():
            if not sv.is_determined:
                sv.sample()
        return -sum(sv.log_prob_sum for sv in process.values() if sv.is_conditioned)
