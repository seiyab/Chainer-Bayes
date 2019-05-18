from chainer_bayes.lazy.lazy_node import LazyNode

class LazyFunction(object):
    def __init__(self, function):
        self.__function = function
    
    @property
    def function(self):
        return self.__function

    def __call__(self, *args, **kwargs):
        return LazyNode(self.function, *args, **kwargs)

def lazify(function):
    return LazyFunction(function)