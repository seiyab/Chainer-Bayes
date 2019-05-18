class LazyNode(object):
    def __init__(self, function, *args, **kwargs):
        self.__function = function
        self.__args = args
        self.__kwargs = kwargs
        self.__value = None

    def evaluate(self):
        if self.is_evaluated:
            return self.__value

        realized_args = [realize(arg) for arg in self.__args]
        realized_kwargs = {key: realize(value) for key, value in self.__kwargs.items()}

        self.__value = self.__function(*realized_args, **realized_kwargs)
        return self.__value

    @property
    def is_evaluated(self):
        return self.__value is not None

    @property
    def dependencies(self):
        return [*self.__args, *self.__kwargs.values()]

def realize(x):
    if issubclass(type(x), LazyNode):
        return x.evaluate()
    else:
        return x