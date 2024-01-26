from datetime import timedelta, datetime
from math import ceil

from pyatf.abort_conditions.abort_condition import AbortCondition
from pyatf.tuning_data import TuningData


class Duration(AbortCondition):
    def __init__(self, duration: timedelta):
        self._duration = duration

    def stop(self, tuning_data: TuningData):
        return tuning_data.total_tuning_duration >= self._duration

    def progress(self, tuning_data: TuningData):
        return min(1.0, (datetime.now() - tuning_data.tuning_start_timestamp) / self._duration)

    def to_json(self):
        microseconds = ceil(self._duration.total_seconds() * 1000000)
        duration_str = (f'{microseconds // 3600000000}'
                        f':{microseconds // 60000000 % 60:02d}'
                        f':{microseconds // 1000000 % 60:02d}'
                        f'.{microseconds % 1000000:06d}')
        return {'kind': 'Duration', 'duration': duration_str}
