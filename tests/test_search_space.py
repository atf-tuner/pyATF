import unittest
from typing import Dict, List, Any, Tuple, Union, Sequence

from pyatf.range import Interval, Set
from pyatf.search_space import Node, SearchSpace, ChainOfTrees
from pyatf.tp import TP


class TestNode(unittest.TestCase):
    def test_init(self):
        d1 = object()
        n = Node(d1)
        self.assertEqual(0, len(n))
        self.assertEqual(0, n.num_leafs)
        self.assertIs(d1, n.data)

    def test_add_child(self):
        d1, d2 = object(), object()

        n = Node(None)
        self.assertEqual(0, len(n))
        self.assertIsNone(n.data)

        c1 = Node(d1)
        n.add_child(c1)
        self.assertEqual(1, len(n))
        self.assertIsNone(n.data)
        self.assertEqual(0, len(c1))
        self.assertIs(d1, c1.data)

        c2 = Node(d2)
        n.add_child(c2)
        self.assertEqual(2, len(n))
        self.assertIsNone(n.data)
        self.assertEqual(0, len(c1))
        self.assertIs(d1, c1.data)
        self.assertEqual(0, len(c2))
        self.assertIs(d2, c2.data)

    def test_get_child(self):
        d1, d2, d3, d4 = object(), object(), object(), object()

        n = Node(None)
        c1 = Node(d1)
        n.add_child(c1)
        c2 = Node(d2)
        n.add_child(c2)
        c3 = Node(d3)
        n.add_child(c3)
        c4 = Node(d4)
        n.add_child(c4)

        self.assertIs(c1, n.get_child(0))
        self.assertIs(c2, n.get_child(1))
        self.assertIs(c3, n.get_child(2))
        self.assertIs(c4, n.get_child(3))


class TestSearchSpace(unittest.TestCase):
    def _check_cot(self, value: ChainOfTrees, gold: List[Tuple[Dict, int]]):
        self.assertEqual(len(gold), len(value))

        def deep_compare(value: Node, gold_num_leafs: int, gold: Sequence[Tuple[Any, Union[Dict, None], int]]):
            self.assertEqual(len(gold), len(value))
            self.assertEqual(gold_num_leafs, value.num_leafs)
            for idx in range(len(value)):
                child = value.get_child(idx)
                gold_data, gold_grandchildren, gold_child_num_leafs = gold[idx]
                self.assertIs(gold_data, child.data)
                self.assertIs(gold_child_num_leafs, child.num_leafs)
                if len(child) > 0 or gold_grandchildren is not None:
                    deep_compare(child, gold_child_num_leafs,
                                 [(data, children, num_leafs)
                                  for data, (children, num_leafs) in gold_grandchildren.items()])

        for value_tree, (gold_tree, gold_num_leafs) in zip(value, gold):
            deep_compare(value_tree.root, gold_num_leafs,
                         [(data, children, num_leafs)
                          for data, (children, num_leafs) in gold_tree.items()])

    def test_single_tp(self):
        tp1 = TP('tp1', Interval(1, 10))
        search_space = SearchSpace(tp1)
        self._check_cot(search_space.cot, [({tp1.values: (None, 1)}, 10)])
        self.assertEqual(10, len(search_space))
        self.assertEqual({'tp1': 1}, search_space.get_configuration((0.00001,)))
        self.assertEqual({'tp1': 1}, search_space.get_configuration((0.10000,)))
        self.assertEqual({'tp1': 8}, search_space.get_configuration((0.70001,)))
        self.assertEqual({'tp1': 8}, search_space.get_configuration((0.72351,)))
        self.assertEqual({'tp1': 8}, search_space.get_configuration((0.80000,)))
        self.assertEqual({'tp1': 10}, search_space.get_configuration((1.00000,)))

    def test_independent_tps(self):
        tp1 = TP('tp1', Interval(1, 10))
        tp2 = TP('tp2', Interval(5, 10))
        search_space = SearchSpace(tp1, tp2)
        self._check_cot(search_space.cot, [({tp1.values: (None, 1)}, 10), ({tp2.values: (None, 1)}, 6)])
        self.assertEqual(60, len(search_space))
        self.assertEqual({'tp1': 1, 'tp2': 5}, search_space.get_configuration((0.00001, 0.00001)))
        self.assertEqual({'tp1': 4, 'tp2': 8}, search_space.get_configuration((0.30001, 0.50001)))
        self.assertEqual({'tp1': 4, 'tp2': 10}, search_space.get_configuration((0.30001, 1.00000)))

    def test_dependent_tps(self):
        search_space = SearchSpace(
            TP('tp1', Interval(1, 10)),
            TP('tp2', Interval(5, 10), lambda tp2, tp1: tp2 % tp1 == 0),
            TP('tp3', Interval(2, 3), lambda tp3, tp1: tp1 % tp3 == 0)
        )
        self._check_cot(search_space.cot, [
            ({
                 2: ({6: ({2: (None, 1)}, 1), 8: ({2: (None, 1)}, 1), 10: ({2: (None, 1)}, 1)}, 3),
                 3: ({6: ({3: (None, 1)}, 1), 9: ({3: (None, 1)}, 1)}, 2),
                 4: ({8: ({2: (None, 1)}, 1)}, 1),
                 6: ({6: ({2: (None, 1), 3: (None, 1)}, 2)}, 2),
                 8: ({8: ({2: (None, 1)}, 1)}, 1),
                 9: ({9: ({3: (None, 1)}, 1)}, 1),
                 10: ({10: ({2: (None, 1)}, 1)}, 1)
             }, 11)
        ])
        self.assertEqual(11, len(search_space))
        self.assertEqual({'tp1': 2, 'tp2': 6, 'tp3': 2}, search_space.get_configuration((0.00001, 0.00001, 0.00001)))
        self.assertEqual({'tp1': 2, 'tp2': 8, 'tp3': 2}, search_space.get_configuration((0.00001, 0.66666, 1.00000)))
        self.assertEqual({'tp1': 6, 'tp2': 6, 'tp3': 2}, search_space.get_configuration((0.60000, 0.00001, 0.50000)))
        self.assertEqual({'tp1': 6, 'tp2': 6, 'tp3': 3}, search_space.get_configuration((0.60000, 1.00000, 0.50001)))

    def test_multiple_dependent_tp_groups(self):
        search_space = SearchSpace(
            TP('tp1', Interval(1, 10)),
            TP('tp2', Interval(5, 10), lambda tp2, tp1: tp2 % tp1 == 0),
            TP('tp3', Interval(2, 3), lambda tp3, tp1: tp1 % tp3 == 0),

            TP('tp4', Set(min, max)),
            TP('tp5', Interval(1, 10)),
            TP('tp6', Interval(1, 10), lambda tp6, tp4, tp5: tp4(tp5, tp6) == 10)
        )
        self._check_cot(search_space.cot, [
            ({
                 2: ({6: ({2: (None, 1)}, 1), 8: ({2: (None, 1)}, 1), 10: ({2: (None, 1)}, 1)}, 3),
                 3: ({6: ({3: (None, 1)}, 1), 9: ({3: (None, 1)}, 1)}, 2),
                 4: ({8: ({2: (None, 1)}, 1)}, 1),
                 6: ({6: ({2: (None, 1), 3: (None, 1)}, 2)}, 2),
                 8: ({8: ({2: (None, 1)}, 1)}, 1),
                 9: ({9: ({3: (None, 1)}, 1)}, 1),
                 10: ({10: ({2: (None, 1)}, 1)}, 1)
             }, 11),
            ({
                 min: ({10: ({10: (None, 1)}, 1)}, 1),
                 max: ({1: ({10: (None, 1)}, 1),
                        2: ({10: (None, 1)}, 1),
                        3: ({10: (None, 1)}, 1),
                        4: ({10: (None, 1)}, 1),
                        5: ({10: (None, 1)}, 1),
                        6: ({10: (None, 1)}, 1),
                        7: ({10: (None, 1)}, 1),
                        8: ({10: (None, 1)}, 1),
                        9: ({10: (None, 1)}, 1),
                        10: ({1: (None, 1), 2: (None, 1), 3: (None, 1), 4: (None, 1), 5: (None, 1),
                              6: (None, 1), 7: (None, 1), 8: (None, 1), 9: (None, 1), 10: (None, 1)}, 10)}, 19)
             }, 20)
        ])
        self.assertEqual(220, len(search_space))
        self.assertEqual({'tp1': 3, 'tp2': 6, 'tp3': 3, 'tp4': min, 'tp5': 10, 'tp6': 10},
                         search_space.get_configuration((0.27273, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001)))
        self.assertEqual({'tp1': 3, 'tp2': 6, 'tp3': 3, 'tp4': min, 'tp5': 10, 'tp6': 10},
                         search_space.get_configuration((0.30000, 0.50000, 1.00000, 0.02334, 0.68413, 0.98321)))
        self.assertEqual({'tp1': 3, 'tp2': 9, 'tp3': 3, 'tp4': min, 'tp5': 10, 'tp6': 10},
                         search_space.get_configuration((0.33333, 0.50001, 0.00001, 0.05000, 1.00000, 1.00000)))
        self.assertEqual({'tp1': 3, 'tp2': 9, 'tp3': 3, 'tp4': max, 'tp5': 1, 'tp6': 10},
                         search_space.get_configuration((0.45454, 0.90000, 1.00000, 0.05001, 0.00001, 0.64537)))
        self.assertEqual({'tp1': 4, 'tp2': 8, 'tp3': 2, 'tp4': max, 'tp5': 9, 'tp6': 10},
                         search_space.get_configuration((0.45455, 0.65410, 0.50000, 0.95348, 0.47368, 0.00001)))
        self.assertEqual({'tp1': 4, 'tp2': 8, 'tp3': 2, 'tp4': max, 'tp5': 10, 'tp6': 7},
                         search_space.get_configuration((0.45455, 0.65410, 0.50001, 1.00000, 0.47369, 0.68753)))

    def test_1d_access(self):
        search_space = SearchSpace(
            TP('tp1', Interval(1, 10)),
            TP('tp2', Interval(5, 10), lambda tp2, tp1: tp2 % tp1 == 0),
            TP('tp3', Interval(2, 3), lambda tp3, tp1: tp1 % tp3 == 0),

            TP('tp4', Set(min, max)),
            TP('tp5', Interval(1, 10)),
            TP('tp6', Interval(1, 10), lambda tp6, tp4, tp5: tp4(tp5, tp6) == 10),

            enable_1d_access=True
        )
        self._check_cot(search_space.cot, [
            ({
                 2: ({6: ({2: (None, 1)}, 1), 8: ({2: (None, 1)}, 1), 10: ({2: (None, 1)}, 1)}, 3),
                 3: ({6: ({3: (None, 1)}, 1), 9: ({3: (None, 1)}, 1)}, 2),
                 4: ({8: ({2: (None, 1)}, 1)}, 1),
                 6: ({6: ({2: (None, 1), 3: (None, 1)}, 2)}, 2),
                 8: ({8: ({2: (None, 1)}, 1)}, 1),
                 9: ({9: ({3: (None, 1)}, 1)}, 1),
                 10: ({10: ({2: (None, 1)}, 1)}, 1)
             }, 11),
            ({
                 min: ({10: ({10: (None, 1)}, 1)}, 1),
                 max: ({1: ({10: (None, 1)}, 1),
                        2: ({10: (None, 1)}, 1),
                        3: ({10: (None, 1)}, 1),
                        4: ({10: (None, 1)}, 1),
                        5: ({10: (None, 1)}, 1),
                        6: ({10: (None, 1)}, 1),
                        7: ({10: (None, 1)}, 1),
                        8: ({10: (None, 1)}, 1),
                        9: ({10: (None, 1)}, 1),
                        10: ({1: (None, 1), 2: (None, 1), 3: (None, 1), 4: (None, 1), 5: (None, 1),
                              6: (None, 1), 7: (None, 1), 8: (None, 1), 9: (None, 1), 10: (None, 1)}, 10)}, 19)
             }, 20)
        ])
        self.assertEqual(220, len(search_space))
        self.assertEqual({'tp1': 2, 'tp2': 6, 'tp3': 2, 'tp4': min, 'tp5': 10, 'tp6': 10},
                         search_space.get_configuration(0))
        self.assertEqual({'tp1': 2, 'tp2': 6, 'tp3': 2, 'tp4': max, 'tp5': 1, 'tp6': 10},
                         search_space.get_configuration(1))
        self.assertEqual({'tp1': 2, 'tp2': 6, 'tp3': 2, 'tp4': max, 'tp5': 2, 'tp6': 10},
                         search_space.get_configuration(2))
        self.assertEqual({'tp1': 2, 'tp2': 8, 'tp3': 2, 'tp4': min, 'tp5': 10, 'tp6': 10},
                         search_space.get_configuration(20))
        self.assertEqual({'tp1': 2, 'tp2': 8, 'tp3': 2, 'tp4': max, 'tp5': 1, 'tp6': 10},
                         search_space.get_configuration(21))
