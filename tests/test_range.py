import unittest

from pyatf.range import Interval, Set


class TestInterval(unittest.TestCase):
    def test_single_element_interval(self):
        r = Interval(3, 3)
        self.assertSequenceEqual([3], list(r))
        self.assertEqual(1, len(r))
        self.assertEqual(3, r[0])

        r = Interval(3, 3, 1)
        self.assertSequenceEqual([3], list(r))
        self.assertEqual(1, len(r))
        self.assertEqual(3, r[0])

        r = Interval(3, 3, 1, lambda i: 2 * i)
        self.assertSequenceEqual([6], list(r))
        self.assertEqual(1, len(r))
        self.assertEqual(6, r[0])

        r = Interval(3.0, 3.0)
        self.assertSequenceEqual([3.0], list(r))
        self.assertEqual(1, len(r))
        self.assertEqual(3.0, r[0])

        r = Interval(3.0, 3.0, 1.0)
        self.assertSequenceEqual([3.0], list(r))
        self.assertEqual(1, len(r))
        self.assertEqual(3.0, r[0])

        r = Interval(3.0, 3.0, 1.0, lambda i: 2.0 * i)
        self.assertSequenceEqual([6.0], list(r))
        self.assertEqual(1, len(r))
        self.assertEqual(6.0, r[0])

    def test_empty_interval(self):
        r = Interval(5, 3)
        self.assertSequenceEqual([], list(r))
        self.assertEqual(0, len(r))

        r = Interval(5, 3, 2)
        self.assertSequenceEqual([], list(r))
        self.assertEqual(0, len(r))

        r = Interval(3, 5, -2, lambda i: 2 * i)
        self.assertSequenceEqual([], list(r))
        self.assertEqual(0, len(r))

        r = Interval(5.0, 3.0)
        self.assertSequenceEqual([], list(r))
        self.assertEqual(0, len(r))

        r = Interval(5.0, 3.0, 2.0)
        self.assertSequenceEqual([], list(r))
        self.assertEqual(0, len(r))

        r = Interval(3.0, 5.0, -2.0, lambda i: 2.0 * i)
        self.assertSequenceEqual([], list(r))
        self.assertEqual(0, len(r))

    def test_multi_element_interval(self):
        r = Interval(3, 5)
        self.assertSequenceEqual([3, 4, 5], list(r))
        self.assertEqual(3, len(r))
        self.assertEqual(3, r[0])
        self.assertEqual(4, r[1])
        self.assertEqual(5, r[2])

        r = Interval(3, 5, 2)
        self.assertSequenceEqual([3, 5], list(r))
        self.assertEqual(2, len(r))
        self.assertEqual(3, r[0])
        self.assertEqual(5, r[1])

        r = Interval(3, 8, 2)
        self.assertSequenceEqual([3, 5, 7], list(r))
        self.assertEqual(3, len(r))
        self.assertEqual(3, r[0])
        self.assertEqual(5, r[1])
        self.assertEqual(7, r[2])

        r = Interval(5, 3, -2)
        self.assertSequenceEqual([5, 3], list(r))
        self.assertEqual(2, len(r))
        self.assertEqual(5, r[0])
        self.assertEqual(3, r[1])

        r = Interval(8, 3, -2)
        self.assertSequenceEqual([8, 6, 4], list(r))
        self.assertEqual(3, len(r))
        self.assertEqual(8, r[0])
        self.assertEqual(6, r[1])
        self.assertEqual(4, r[2])

        r = Interval(3.0, 5.0)
        self.assertSequenceEqual([3.0, 4.0, 5.0], list(r))
        self.assertEqual(3, len(r))
        self.assertEqual(3.0, r[0])
        self.assertEqual(4.0, r[1])
        self.assertEqual(5.0, r[2])

        r = Interval(3.0, 5.0, 0.5)
        self.assertSequenceEqual([3.0, 3.5, 4.0, 4.5, 5.0], list(r))
        self.assertEqual(5, len(r))
        self.assertEqual(3.0, r[0])
        self.assertEqual(3.5, r[1])
        self.assertEqual(4.0, r[2])
        self.assertEqual(4.5, r[3])
        self.assertEqual(5.0, r[4])

        r = Interval(3.0, 5.0, 0.3)
        self.assertSequenceEqual([3.0, 3.3, 3.6, 3.9, 4.2, 4.5, 4.8], list(r))
        self.assertEqual(7, len(r))
        self.assertEqual(3.0, r[0])
        self.assertEqual(3.3, r[1])
        self.assertEqual(3.6, r[2])
        self.assertEqual(3.9, r[3])
        self.assertEqual(4.2, r[4])
        self.assertEqual(4.5, r[5])
        self.assertEqual(4.8, r[6])

        r = Interval(5.0, 3.0, -0.5)
        self.assertSequenceEqual([5.0, 4.5, 4.0, 3.5, 3.0], list(r))
        self.assertEqual(5, len(r))
        self.assertEqual(5.0, r[0])
        self.assertEqual(4.5, r[1])
        self.assertEqual(4.0, r[2])
        self.assertEqual(3.5, r[3])
        self.assertEqual(3.0, r[4])

        r = Interval(5.0, 3.0, -0.3)
        self.assertSequenceEqual([5.0, 4.7, 4.4, 4.1, 3.8, 3.5, 3.2], list(r))
        self.assertEqual(7, len(r))
        self.assertEqual(5.0, r[0])
        self.assertEqual(4.7, r[1])
        self.assertEqual(4.4, r[2])
        self.assertEqual(4.1, r[3])
        self.assertEqual(3.8, r[4])
        self.assertEqual(3.5, r[5])
        self.assertEqual(3.2, r[6])

    def test_multiple_iterations(self):
        r = Interval(3, 5)
        self.assertSequenceEqual([3, 4, 5], list(r))
        self.assertSequenceEqual([3, 4, 5], list(r))


class TestSet(unittest.TestCase):
    def test_empty_set(self):
        s = Set()
        self.assertSequenceEqual([], list(s))
        self.assertEqual(0, len(s))

    def test_single_element_set(self):
        s = Set('val1')
        self.assertSequenceEqual(['val1'], list(s))
        self.assertEqual(1, len(s))
        self.assertEqual('val1', s[0])

    def test_multi_element_set(self):
        s = Set('val1', 'val2', 'val3')
        self.assertSequenceEqual(['val1', 'val2', 'val3'], list(s))
        self.assertEqual(3, len(s))
        self.assertEqual('val1', s[0])
        self.assertEqual('val2', s[1])
        self.assertEqual('val3', s[2])

    def test_multiple_iterations(self):
        s = Set('val1', 'val2')
        self.assertSequenceEqual(['val1', 'val2'], list(s))
        self.assertSequenceEqual(['val1', 'val2'], list(s))
