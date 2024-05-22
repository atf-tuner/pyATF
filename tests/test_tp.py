import unittest

from pyatf import TP
from pyatf.range import Interval


class TestTP(unittest.TestCase):
    def test_get_lambda_source(self):
        tp1 = TP('tp1', Interval(1, 10), lambda tp1: True)
        self.assertEqual({
            'name': 'tp1',
            'range': {'kind': 'Interval', 'start': 1, 'end': 10, 'step': 1},
            'constraint': '''tp1 = TP('tp1', Interval(1, 10), lambda tp1: True)\n'''
        }, tp1.to_json())

    def test_get_func_source(self):
        def tp1_constraint(tp1):
            return True
        tp1 = TP('tp1', Interval(1, 10), tp1_constraint)
        self.assertEqual({
            'name': 'tp1',
            'range': {'kind': 'Interval', 'start': 1, 'end': 10, 'step': 1},
            'constraint': '''def tp1_constraint(tp1):\n    return True\n'''
        }, tp1.to_json())

    def test_get_eval_lambda_source(self):
        constraint_source = 'lambda tp1: True'
        tp1 = TP('tp1', Interval(1, 10), eval(constraint_source))
        self.assertEqual({
            'name': 'tp1',
            'range': {'kind': 'Interval', 'start': 1, 'end': 10, 'step': 1},
            'constraint': '''source unknown'''
        }, tp1.to_json())
        tp1.constraint_source = constraint_source
        self.assertEqual({
            'name': 'tp1',
            'range': {'kind': 'Interval', 'start': 1, 'end': 10, 'step': 1},
            'constraint': '''lambda tp1: True'''
        }, tp1.to_json())
