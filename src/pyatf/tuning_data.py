import copy
import time
from datetime import timedelta, datetime
from math import ceil
from typing import Any, Dict, Optional, Tuple, Callable, List, Union

Configuration = Dict[str, Any]

Cost = float
MetaData = Dict[str, Any]

# CostFunction returns Cost for a given Configuration and optional meta-data for logging purposes as a dictionary
CostFunction = Callable[[Configuration], Union[Cost, Tuple[Cost, MetaData]]]


# Error to be thrown in cost functions, when additional meta-data was collected
class CostFunctionError(BaseException):
    def __init__(self, meta_data: MetaData):
        self.meta_data: MetaData = copy.deepcopy(meta_data)


Coordinates = Tuple[float, ...]
Index = int


class History:
    class Entry:
        def __init__(self, timestamp: datetime, timedelta_since_tuning_start: timedelta,
                     evaluations: int, valid_evaluations: int,
                     configuration: Configuration, valid: bool, cost: Optional[Cost] = None,
                     meta_data: Optional[Dict] = None,
                     search_space_coordinates: Optional[Coordinates] = None,
                     search_space_index: Optional[Index] = None):
            self._timestamp = timestamp
            self._timedelta_since_tuning_start = timedelta_since_tuning_start
            self._evaluations = evaluations
            self._valid_evaluations = valid_evaluations
            self._configuration = configuration.copy()
            self._valid = valid
            self._cost = cost
            self._meta_data = copy.deepcopy(meta_data)
            self._search_space_coordinates = search_space_coordinates
            self._search_space_index = search_space_index

        @property
        def timestamp(self):
            return self._timestamp

        @property
        def timedelta_since_tuning_start(self):
            return self._timedelta_since_tuning_start

        @property
        def evaluations(self):
            return self._evaluations

        @property
        def valid_evaluations(self):
            return self._valid_evaluations

        @property
        def configuration(self):
            return self._configuration

        @property
        def valid(self):
            return self._valid

        @property
        def cost(self):
            return self._cost

        @property
        def meta_data(self):
            return self._meta_data

        @property
        def search_space_coordinates(self):
            return self._search_space_coordinates

        @property
        def search_space_index(self):
            return self._search_space_index

        def to_json(self):
            microseconds = ceil(self.timedelta_since_tuning_start.total_seconds() * 1000000)
            timedelta_since_tuning_start_str = (f'{microseconds // 3600000000}'
                                                f':{microseconds // 60000000 % 60:02d}'
                                                f':{microseconds // 1000000 % 60:02d}'
                                                f'.{microseconds % 1000000:06d}')
            json = {
                'timestamp': self._timestamp.strftime('%Y-%m-%dT%H:%M:%S.%f'),
                'timedelta_since_tuning_start': timedelta_since_tuning_start_str,
                'evaluations': self._evaluations,
                'valid_evaluations': self._valid_evaluations,
                'configuration': copy.deepcopy(self._configuration),
                'valid': self._valid,
                'cost': self._cost
            }
            if self._meta_data is not None:
                json['meta_data'] = copy.deepcopy(self._meta_data)
            if self._search_space_coordinates is not None:
                json['search_space_coordinates'] = self._search_space_coordinates
            if self._search_space_index is not None:
                json['search_space_index'] = self._search_space_index
            return json

    def __init__(self):
        self._entries = []

    def append(self, entry: Entry):
        self._entries.append(entry)

    def is_empty(self):
        return not self._entries

    def __getitem__(self, item):
        return self._entries[item]

    def __iter__(self):
        yield from self._entries

    def __len__(self):
        return len(self._entries)

    def to_json(self):
        return list(entry.to_json() for entry in self._entries)


class TuningData:
    def __init__(self,
                 tuning_parameters: List[Dict],
                 constrained_search_space_size: int,
                 unconstrained_search_space_size: int,
                 search_space_generation_ns: int,
                 search_technique: Dict,
                 abort_condition: Dict):
        self.tuning_parameters = copy.deepcopy(tuning_parameters)
        self.constrained_search_space_size = constrained_search_space_size
        self.unconstrained_search_space_size = unconstrained_search_space_size
        self.search_space_generation_ns = search_space_generation_ns
        self.search_technique = copy.deepcopy(search_technique)
        self.abort_condition = copy.deepcopy(abort_condition)
        self.tuning_start_timestamp = datetime.now()
        self._tuning_start_perf_counter = time.perf_counter_ns()
        self._total_tuning_duration: Optional[timedelta] = None
        self.terminated_early: bool = False
        self.history: History = History()
        self.improvement_history: History = History()
        self.number_of_evaluated_configurations: int = 0
        self.number_of_evaluated_valid_configurations: int = 0
        self.number_of_evaluated_invalid_configurations: int = 0

    @property
    def total_tuning_duration(self):
        if self._total_tuning_duration is None:
            now = self.tuning_start_timestamp + timedelta(
                microseconds=(time.perf_counter_ns() - self._tuning_start_perf_counter) / 1000
            )
            return now - self.tuning_start_timestamp
        else:
            return self._total_tuning_duration

    def min_cost(self):
        if self.improvement_history.is_empty():
            return None
        else:
            return self.improvement_history[-1].cost

    def configuration_of_min_cost(self):
        if self.improvement_history.is_empty():
            return None
        else:
            return self.improvement_history[-1].configuration

    def meta_data_of_min_cost(self):
        if self.improvement_history.is_empty():
            return None
        else:
            return self.improvement_history[-1].meta_data

    def search_space_coordinates_of_min_cost(self):
        if self.improvement_history.is_empty():
            return None
        else:
            return self.improvement_history[-1].search_space_coordinates

    def search_space_index_of_min_cost(self):
        if self.improvement_history.is_empty():
            return None
        else:
            return self.improvement_history[-1].search_space_index

    def timestamp_of_min_cost(self):
        if self.improvement_history.is_empty():
            return None
        else:
            return self.improvement_history[-1].timestamp

    def duration_to_min_cost(self):
        if self.improvement_history.is_empty():
            return None
        else:
            return self.improvement_history[-1].timedelta_since_tuning_start

    def evaluations_to_min_cost(self):
        if self.improvement_history.is_empty():
            return None
        else:
            return self.improvement_history[-1].evaluations

    def valid_evaluations_to_min_cost(self):
        if self.improvement_history.is_empty():
            return None
        else:
            return self.improvement_history[-1].valid_evaluations

    def record_evaluation(self, configuration: Configuration, valid: bool, cost: Optional[Cost] = None,
                          meta_data: Optional[Dict] = None,
                          search_space_coordinates: Optional[Coordinates] = None,
                          search_space_index: Optional[Index] = None) -> datetime:
        now = self.tuning_start_timestamp + timedelta(
            microseconds=(time.perf_counter_ns() - self._tuning_start_perf_counter) / 1000
        )
        if valid and cost is None:
            raise ValueError('expecting cost if valid is True')
        if self._total_tuning_duration is not None:
            raise ValueError('cannot record evaluations after tuning finish has been recorded')

        self.number_of_evaluated_configurations += 1
        if valid:
            self.number_of_evaluated_valid_configurations += 1
        else:
            self.number_of_evaluated_invalid_configurations += 1
        entry = History.Entry(
            now,
            now - self.tuning_start_timestamp,
            self.number_of_evaluated_configurations,
            self.number_of_evaluated_valid_configurations,
            configuration,
            valid,
            cost,
            meta_data,
            search_space_coordinates,
            search_space_index
        )
        self.history.append(entry)
        new_best = valid and (self.improvement_history.is_empty() or cost < self.improvement_history[-1].cost)
        if new_best:
            self.improvement_history.append(entry)
        return now

    def record_tuning_finished(self, terminated_early: bool):
        now = self.tuning_start_timestamp + timedelta(
            microseconds=(time.perf_counter_ns() - self._tuning_start_perf_counter) / 1000
        )
        self._total_tuning_duration = now - self.tuning_start_timestamp
        self.terminated_early = terminated_early

    def to_json(self):
        total_tuning_microseconds = ceil(self.total_tuning_duration.total_seconds() * 1000000)
        total_tuning_duration_str = (f'{total_tuning_microseconds // 3600000000}'
                                     f':{total_tuning_microseconds // 60000000 % 60:02d}'
                                     f':{total_tuning_microseconds // 1000000 % 60:02d}'
                                     f'.{total_tuning_microseconds % 1000000:06d}')
        duration_to_min_cost_str = None
        if self.duration_to_min_cost() is not None:
            microseconds_to_min_cost = ceil(self.duration_to_min_cost().total_seconds() * 1000000)
            duration_to_min_cost_str = (f'{microseconds_to_min_cost // 3600000000}'
                                        f':{microseconds_to_min_cost // 60000000 % 60:02d}'
                                        f':{microseconds_to_min_cost // 1000000 % 60:02d}'
                                        f'.{microseconds_to_min_cost % 1000000:06d}')
        timestamp_of_min_cost_str = None
        if self.timestamp_of_min_cost() is not None:
            timestamp_of_min_cost_str = self.timestamp_of_min_cost().strftime('%Y-%m-%dT%H:%M:%S.%f')
        return {
            'tuning_parameters': copy.deepcopy(self.tuning_parameters),
            'constrained_search_space_size': self.constrained_search_space_size,
            'unconstrained_search_space_size': self.unconstrained_search_space_size,
            'search_space_generation_ns': self.search_space_generation_ns,
            'search_technique': copy.deepcopy(self.search_technique),
            'abort_condition': copy.deepcopy(self.abort_condition),
            'tuning_start_timestamp': self.tuning_start_timestamp.strftime('%Y-%m-%dT%H:%M:%S.%f'),
            'total_tuning_duration': total_tuning_duration_str,
            'terminated_early': self.terminated_early,
            'history': self.history.to_json(),
            'improvement_history': self.improvement_history.to_json(),
            'number_of_evaluated_configurations': self.number_of_evaluated_configurations,
            'number_of_evaluated_valid_configurations': self.number_of_evaluated_valid_configurations,
            'number_of_evaluated_invalid_configurations': self.number_of_evaluated_invalid_configurations,
            'min_cost': self.min_cost(),
            'configuration_of_min_cost': copy.deepcopy(self.configuration_of_min_cost()),
            'meta_data_of_min_cost': copy.deepcopy(self.meta_data_of_min_cost()),
            'search_space_coordinates_of_min_cost': self.search_space_coordinates_of_min_cost(),
            'search_space_index_of_min_cost': self.search_space_index_of_min_cost(),
            'timestamp_of_min_cost': timestamp_of_min_cost_str,
            'duration_to_min_cost': duration_to_min_cost_str,
            'evaluations_to_min_cost': self.evaluations_to_min_cost(),
            'valid_evaluations_to_min_cost': self.valid_evaluations_to_min_cost()
        }
