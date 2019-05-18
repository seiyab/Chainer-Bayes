import unittest

from operator import add
from operator import sub

from chainer_bayes import lazy

class TestLazyNode(unittest.TestCase):
    def test_evaluate(self):
        lazy_node = lazy.LazyNode(add, 1, 2)
        
        self.assertEqual(lazy_node.evaluate(), 3)

    def test_two_steps(self):
        lazy_five = lazy.LazyNode(add, 2, 3)
        lazy_four = lazy.LazyNode(sub, lazy_five, 1)

        self.assertEqual(lazy_four.evaluate(), 4)
        self.assertTrue(lazy_five.is_evaluated)

