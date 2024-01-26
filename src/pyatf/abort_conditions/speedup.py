from datetime import timedelta, datetime
from math import ceil
from typing import Optional

from pyatf.abort_conditions.abort_condition import AbortCondition
from pyatf.tuning_data import TuningData


class Speedup(AbortCondition):
    def __init__(self, speedup: float,
                 time: Optional[timedelta] = None,
                 evaluations: Optional[int] = None,
                 valid_evaluations: Optional[int] = None):
        num_interval_args = 0
        if time is not None:
            num_interval_args += 1
        if evaluations is not None:
            num_interval_args += 1
        if valid_evaluations is not None:
            num_interval_args += 1
        if num_interval_args != 1:
            raise ValueError('expecting exactly one interval specification '
                             '(either time, evaluations, or valid_evaluations)')

        self._min_speedup = speedup
        self._time = time
        self._evaluations = evaluations
        self._valid_evaluations = valid_evaluations

    def stop(self, tuning_data: TuningData):
        speedup = 0
        if self._time is not None:
            if tuning_data.improvement_history.is_empty():
                return (datetime.now() - tuning_data.tuning_start_timestamp) >= self._time
            else:
                best_result = tuning_data.min_cost()
                cutoff = datetime.now() - self._time
                for entry in reversed(tuning_data.improvement_history):
                    if entry.timestamp < cutoff:
                        break
                    speedup = entry.cost / best_result
        elif self._evaluations is not None:
            if tuning_data.improvement_history.is_empty():
                return tuning_data.number_of_evaluated_configurations >= self._evaluations
            else:
                best_result = tuning_data.min_cost()
                cutoff = tuning_data.number_of_evaluated_configurations - self._evaluations
                for entry in reversed(tuning_data.improvement_history):
                    if entry.evaluations < cutoff:
                        break
                    speedup = entry.cost / best_result
        elif self._valid_evaluations is not None:
            if tuning_data.improvement_history.is_empty():
                return tuning_data.number_of_evaluated_valid_configurations >= self._valid_evaluations
            else:
                best_result = tuning_data.min_cost()
                cutoff = tuning_data.number_of_evaluated_valid_configurations - self._valid_evaluations
                for entry in reversed(tuning_data.improvement_history):
                    if entry.valid_evaluations < cutoff:
                        break
                    speedup = entry.cost / best_result
        return speedup >= self._min_speedup

    def to_json(self):
        json = {'kind': 'Speedup', 'speedup': self._min_speedup}
        if self._time is not None:
            microseconds = ceil(self._time.total_seconds() * 1000000)
            time_str = (f'{microseconds // 3600000000}'
                        f':{microseconds // 60000000 % 60:02d}'
                        f':{microseconds // 1000000 % 60:02d}'
                        f'.{microseconds % 1000000:06d}')
            json['time'] = time_str
        elif self._evaluations is not None:
            json['evaluations'] = self._evaluations
        elif self._valid_evaluations is not None:
            json['valid_evaluations'] = self._valid_evaluations
        return json
