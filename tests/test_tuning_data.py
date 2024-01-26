import time
import unittest
from datetime import datetime
from typing import Tuple, Optional, Dict

from pyatf.abort_conditions import Evaluations
from pyatf.range import Interval
from pyatf.search_space import SearchSpace
from pyatf.search_techniques import Random
from pyatf.tp import TP
from pyatf.tuning_data import TuningData, Configuration, Cost, History, Coordinates, Index


class TestTuningData(unittest.TestCase):
    def _check_history(self,
                       gold_history: Tuple[Tuple[datetime, int, int, Configuration, bool, Optional[Cost],
                       Optional[Dict], Optional[Coordinates], Optional[Index]], ...],
                       tuning_start: datetime,
                       history: History):
        self.assertEqual(len(gold_history), len(history))
        for record, gold_record in zip(history, gold_history):
            gold_record_timestamp = gold_record[0]
            gold_record_num_evaluations = gold_record[1]
            gold_record_num_valid_evaluations = gold_record[2]
            gold_record_configuration = gold_record[3]
            gold_record_valid = gold_record[4]
            gold_record_cost = gold_record[5]
            gold_record_meta_data = gold_record[6]
            gold_record_coordinates = gold_record[7]
            gold_record_index = gold_record[8]
            self.assertEqual(gold_record_timestamp, record.timestamp)
            self.assertEqual(record.timestamp - tuning_start, record.timedelta_since_tuning_start)
            self.assertEqual(gold_record_num_evaluations, record.evaluations)
            self.assertEqual(gold_record_num_valid_evaluations, record.valid_evaluations)
            self.assertEqual(gold_record_configuration, record.configuration)
            self.assertEqual(gold_record_valid, record.valid)
            self.assertEqual(gold_record_cost, record.cost)
            self.assertEqual(gold_record_meta_data, record.meta_data)
            self.assertEqual(gold_record_coordinates, record.search_space_coordinates)
            self.assertEqual(gold_record_index, record.search_space_index)

    def test(self):
        tp1 = TP('tp1', Interval(1, 10))
        tp2 = TP('tp2', Interval(5, 10), lambda tp2, tp1: tp2 % tp1 == 0)
        tp3 = TP('tp3', Interval(2, 3), lambda tp3, tp1: tp1 % tp3 == 0)
        search_space_generation_start = time.perf_counter_ns()
        search_space = SearchSpace(tp1, tp2, tp3, enable_1d_access=True)
        search_space_generation_end = time.perf_counter_ns()
        tuning_data = TuningData(
            [tp1.to_json(), tp2.to_json(), tp3.to_json()],
            len(search_space),
            len(tp1.values) * len(tp2.values) * len(tp3.values),
            search_space_generation_end - search_space_generation_start,
            Random().to_json(),
            Evaluations(6).to_json()
        )

        self.assertEqual([
            {'name': 'tp1', 'range': {'kind': 'Interval', 'start': 1, 'end': 10, 'step': 1}},
            {'name': 'tp2', 'range': {'kind': 'Interval', 'start': 5, 'end': 10, 'step': 1},
             'constraint': '        tp2 = TP(\'tp2\', Interval(5, 10), lambda tp2, tp1: tp2 % tp1 == 0)\n'},
            {'name': 'tp3', 'range': {'kind': 'Interval', 'start': 2, 'end': 3, 'step': 1},
             'constraint': '        tp3 = TP(\'tp3\', Interval(2, 3), lambda tp3, tp1: tp1 % tp3 == 0)\n'}
        ], tuning_data.tuning_parameters)
        self.assertEqual(11, tuning_data.constrained_search_space_size)
        self.assertEqual(120, tuning_data.unconstrained_search_space_size)
        self.assertEqual(search_space_generation_end - search_space_generation_start,
                         tuning_data.search_space_generation_ns)
        self.assertEqual({'kind': 'Random'}, tuning_data.search_technique)
        self.assertEqual({'kind': 'Evaluations', 'evaluations': 6}, tuning_data.abort_condition)
        self._check_history(tuple(), tuning_data.tuning_start_timestamp, tuning_data.history)
        self._check_history(tuple(), tuning_data.tuning_start_timestamp, tuning_data.improvement_history)
        self.assertEqual(0, tuning_data.number_of_evaluated_configurations)
        self.assertEqual(0, tuning_data.number_of_evaluated_valid_configurations)
        self.assertEqual(0, tuning_data.number_of_evaluated_invalid_configurations)
        self.assertIsNone(tuning_data.min_cost())
        self.assertIsNone(tuning_data.configuration_of_min_cost())
        self.assertIsNone(tuning_data.meta_data_of_min_cost())
        self.assertIsNone(tuning_data.search_space_coordinates_of_min_cost())
        self.assertIsNone(tuning_data.search_space_index_of_min_cost())
        self.assertIsNone(tuning_data.timestamp_of_min_cost())
        self.assertIsNone(tuning_data.duration_to_min_cost())
        self.assertIsNone(tuning_data.evaluations_to_min_cost())
        self.assertIsNone(tuning_data.valid_evaluations_to_min_cost())

        coords1 = (0.12938, 0.83746, 0.91349)
        index1 = None
        config1 = search_space.get_configuration(coords1 or index1)
        valid1 = False
        cost1 = None
        meta_data1 = {'error_code': -4}
        timestamp1 = tuning_data.record_evaluation(config1, valid1, cost1, meta_data1, coords1, index1)
        self._check_history((
            (timestamp1, 1, 0, config1, valid1, cost1, meta_data1, coords1, index1),
        ), tuning_data.tuning_start_timestamp, tuning_data.history)
        self._check_history(tuple(), tuning_data.tuning_start_timestamp, tuning_data.improvement_history)
        self.assertEqual(1, tuning_data.number_of_evaluated_configurations)
        self.assertEqual(0, tuning_data.number_of_evaluated_valid_configurations)
        self.assertEqual(1, tuning_data.number_of_evaluated_invalid_configurations)
        self.assertIsNone(tuning_data.min_cost())
        self.assertIsNone(tuning_data.configuration_of_min_cost())
        self.assertIsNone(tuning_data.meta_data_of_min_cost())
        self.assertIsNone(tuning_data.search_space_coordinates_of_min_cost())
        self.assertIsNone(tuning_data.search_space_index_of_min_cost())
        self.assertIsNone(tuning_data.timestamp_of_min_cost())
        self.assertIsNone(tuning_data.duration_to_min_cost())
        self.assertIsNone(tuning_data.evaluations_to_min_cost())
        self.assertIsNone(tuning_data.valid_evaluations_to_min_cost())

        coords2 = None
        index2 = 3
        config2 = search_space.get_configuration(coords2 or index2)
        valid2 = False
        cost2 = None
        meta_data2 = {'error_code': -6}
        timestamp2 = tuning_data.record_evaluation(config2, valid2, cost2, meta_data2, coords2, index2)
        self._check_history((
            (timestamp1, 1, 0, config1, valid1, cost1, meta_data1, coords1, index1),
            (timestamp2, 2, 0, config2, valid2, cost2, meta_data2, coords2, index2),
        ), tuning_data.tuning_start_timestamp, tuning_data.history)
        self._check_history(tuple(), tuning_data.tuning_start_timestamp, tuning_data.improvement_history)
        self.assertEqual(2, tuning_data.number_of_evaluated_configurations)
        self.assertEqual(0, tuning_data.number_of_evaluated_valid_configurations)
        self.assertEqual(2, tuning_data.number_of_evaluated_invalid_configurations)
        self.assertIsNone(tuning_data.min_cost())
        self.assertIsNone(tuning_data.configuration_of_min_cost())
        self.assertIsNone(tuning_data.meta_data_of_min_cost())
        self.assertIsNone(tuning_data.search_space_coordinates_of_min_cost())
        self.assertIsNone(tuning_data.search_space_index_of_min_cost())
        self.assertIsNone(tuning_data.timestamp_of_min_cost())
        self.assertIsNone(tuning_data.duration_to_min_cost())
        self.assertIsNone(tuning_data.evaluations_to_min_cost())
        self.assertIsNone(tuning_data.valid_evaluations_to_min_cost())

        coords3 = (0.234124, 0.123123, 0.5382921)
        index3 = None
        config3 = search_space.get_configuration(coords3 or index3)
        valid3 = True
        cost3 = 10.23728
        meta_data3 = None
        timestamp3 = tuning_data.record_evaluation(config3, valid3, cost3, meta_data3, coords3, index3)
        self._check_history((
            (timestamp1, 1, 0, config1, valid1, cost1, meta_data1, coords1, index1),
            (timestamp2, 2, 0, config2, valid2, cost2, meta_data2, coords2, index2),
            (timestamp3, 3, 1, config3, valid3, cost3, meta_data3, coords3, index3),
        ), tuning_data.tuning_start_timestamp, tuning_data.history)
        self._check_history((
            (timestamp3, 3, 1, config3, valid3, cost3, meta_data3, coords3, index3),
        ), tuning_data.tuning_start_timestamp, tuning_data.improvement_history)
        self.assertEqual(3, tuning_data.number_of_evaluated_configurations)
        self.assertEqual(1, tuning_data.number_of_evaluated_valid_configurations)
        self.assertEqual(2, tuning_data.number_of_evaluated_invalid_configurations)
        self.assertEqual(cost3, tuning_data.min_cost())
        self.assertEqual(config3, tuning_data.configuration_of_min_cost())
        self.assertEqual(meta_data3, tuning_data.meta_data_of_min_cost())
        self.assertEqual(coords3, tuning_data.search_space_coordinates_of_min_cost())
        self.assertEqual(index3, tuning_data.search_space_index_of_min_cost())
        self.assertEqual(timestamp3, tuning_data.timestamp_of_min_cost())
        self.assertEqual(timestamp3 - tuning_data.tuning_start_timestamp, tuning_data.duration_to_min_cost())
        self.assertEqual(3, tuning_data.evaluations_to_min_cost())
        self.assertEqual(1, tuning_data.valid_evaluations_to_min_cost())

        coords4 = (0.49857, 0.9813284, 0.757172)
        index4 = None
        config4 = search_space.get_configuration(coords4 or index4)
        valid4 = False
        cost4 = None
        meta_data4 = None
        timestamp4 = tuning_data.record_evaluation(config4, valid4, cost4, meta_data4, coords4, index4)
        self._check_history((
            (timestamp1, 1, 0, config1, valid1, cost1, meta_data1, coords1, index1),
            (timestamp2, 2, 0, config2, valid2, cost2, meta_data2, coords2, index2),
            (timestamp3, 3, 1, config3, valid3, cost3, meta_data3, coords3, index3),
            (timestamp4, 4, 1, config4, valid4, cost4, meta_data4, coords4, index4),
        ), tuning_data.tuning_start_timestamp, tuning_data.history)
        self._check_history((
            (timestamp3, 3, 1, config3, valid3, cost3, meta_data3, coords3, index3),
        ), tuning_data.tuning_start_timestamp, tuning_data.improvement_history)
        self.assertEqual(4, tuning_data.number_of_evaluated_configurations)
        self.assertEqual(1, tuning_data.number_of_evaluated_valid_configurations)
        self.assertEqual(3, tuning_data.number_of_evaluated_invalid_configurations)
        self.assertEqual(cost3, tuning_data.min_cost())
        self.assertEqual(config3, tuning_data.configuration_of_min_cost())
        self.assertEqual(meta_data3, tuning_data.meta_data_of_min_cost())
        self.assertEqual(coords3, tuning_data.search_space_coordinates_of_min_cost())
        self.assertEqual(index3, tuning_data.search_space_index_of_min_cost())
        self.assertEqual(timestamp3, tuning_data.timestamp_of_min_cost())
        self.assertEqual(timestamp3 - tuning_data.tuning_start_timestamp, tuning_data.duration_to_min_cost())
        self.assertEqual(3, tuning_data.evaluations_to_min_cost())
        self.assertEqual(1, tuning_data.valid_evaluations_to_min_cost())

        coords5 = None
        index5 = 10
        config5 = search_space.get_configuration(coords5 or index5)
        valid5 = True
        cost5 = 8.46543
        meta_data5 = {'compile_time': 929.224}
        timestamp5 = tuning_data.record_evaluation(config5, valid5, cost5, meta_data5, coords5, index5)
        self._check_history((
            (timestamp1, 1, 0, config1, valid1, cost1, meta_data1, coords1, index1),
            (timestamp2, 2, 0, config2, valid2, cost2, meta_data2, coords2, index2),
            (timestamp3, 3, 1, config3, valid3, cost3, meta_data3, coords3, index3),
            (timestamp4, 4, 1, config4, valid4, cost4, meta_data4, coords4, index4),
            (timestamp5, 5, 2, config5, valid5, cost5, meta_data5, coords5, index5),
        ), tuning_data.tuning_start_timestamp, tuning_data.history)
        self._check_history((
            (timestamp3, 3, 1, config3, valid3, cost3, meta_data3, coords3, index3),
            (timestamp5, 5, 2, config5, valid5, cost5, meta_data5, coords5, index5),
        ), tuning_data.tuning_start_timestamp, tuning_data.improvement_history)
        self.assertEqual(5, tuning_data.number_of_evaluated_configurations)
        self.assertEqual(2, tuning_data.number_of_evaluated_valid_configurations)
        self.assertEqual(3, tuning_data.number_of_evaluated_invalid_configurations)
        self.assertEqual(cost5, tuning_data.min_cost())
        self.assertEqual(config5, tuning_data.configuration_of_min_cost())
        self.assertEqual(meta_data5, tuning_data.meta_data_of_min_cost())
        self.assertEqual(coords5, tuning_data.search_space_coordinates_of_min_cost())
        self.assertEqual(index5, tuning_data.search_space_index_of_min_cost())
        self.assertEqual(timestamp5, tuning_data.timestamp_of_min_cost())
        self.assertEqual(timestamp5 - tuning_data.tuning_start_timestamp, tuning_data.duration_to_min_cost())
        self.assertEqual(5, tuning_data.evaluations_to_min_cost())
        self.assertEqual(2, tuning_data.valid_evaluations_to_min_cost())

        coords6 = (0.298382, 1.00000, 0.263831)
        index6 = None
        config6 = search_space.get_configuration(coords6 or index6)
        valid6 = True
        cost6 = 9.78224
        meta_data6 = {'compile_time': 55.125}
        timestamp6 = tuning_data.record_evaluation(config6, valid6, cost6, meta_data6, coords6, index6)
        self._check_history((
            (timestamp1, 1, 0, config1, valid1, cost1, meta_data1, coords1, index1),
            (timestamp2, 2, 0, config2, valid2, cost2, meta_data2, coords2, index2),
            (timestamp3, 3, 1, config3, valid3, cost3, meta_data3, coords3, index3),
            (timestamp4, 4, 1, config4, valid4, cost4, meta_data4, coords4, index4),
            (timestamp5, 5, 2, config5, valid5, cost5, meta_data5, coords5, index5),
            (timestamp6, 6, 3, config6, valid6, cost6, meta_data6, coords6, index6),
        ), tuning_data.tuning_start_timestamp, tuning_data.history)
        self._check_history((
            (timestamp3, 3, 1, config3, valid3, cost3, meta_data3, coords3, index3),
            (timestamp5, 5, 2, config5, valid5, cost5, meta_data5, coords5, index5),
        ), tuning_data.tuning_start_timestamp, tuning_data.improvement_history)
        self.assertEqual(6, tuning_data.number_of_evaluated_configurations)
        self.assertEqual(3, tuning_data.number_of_evaluated_valid_configurations)
        self.assertEqual(3, tuning_data.number_of_evaluated_invalid_configurations)
        self.assertEqual(cost5, tuning_data.min_cost())
        self.assertEqual(config5, tuning_data.configuration_of_min_cost())
        self.assertEqual(meta_data5, tuning_data.meta_data_of_min_cost())
        self.assertEqual(coords5, tuning_data.search_space_coordinates_of_min_cost())
        self.assertEqual(index5, tuning_data.search_space_index_of_min_cost())
        self.assertEqual(timestamp5, tuning_data.timestamp_of_min_cost())
        self.assertEqual(timestamp5 - tuning_data.tuning_start_timestamp, tuning_data.duration_to_min_cost())
        self.assertEqual(5, tuning_data.evaluations_to_min_cost())
        self.assertEqual(2, tuning_data.valid_evaluations_to_min_cost())

        tuning_data.record_tuning_finished(False)
