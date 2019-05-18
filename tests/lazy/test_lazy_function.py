import unittest

from operator import add
from operator import sub

from chainer_bayes import lazy

class TestLazyNode(unittest.TestCase):
    def test_lazify(self):
        lazy_add = lazy.lazify(add)
        lazy_three = lazy_add(2, 1)

        self.assertIsInstance(lazy_add, lazy.LazyFunction)
        self.assertIsInstance(lazy_three, lazy.LazyNode)
        self.assertEqual(lazy_add.function, add)
        self.assertEqual(lazy_three.evaluate(), 3)