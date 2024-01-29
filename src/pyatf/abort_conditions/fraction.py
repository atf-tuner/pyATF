from pyatf.abort_conditions.abort_condition import AbortCondition
from pyatf.tuning_data import TuningData


class Fraction(AbortCondition):
    def __init__(self, fraction: float):
        self._fraction = fraction

    def stop(self, tuning_data: TuningData):
        return (tuning_data.number_of_evaluated_configurations >=
                self._fraction * tuning_data.constrained_search_space_size)

    def progress(self, tuning_data: TuningData):
        return (tuning_data.number_of_evaluated_configurations
                / tuning_data.constrained_search_space_size
                / self._fraction)

    def to_json(self):
        return {'kind': 'Fraction', 'fraction': self._fraction}


class ValidFraction(AbortCondition):
    def __init__(self, fraction: float):
        self._fraction = fraction

    def stop(self, tuning_data: TuningData):
        return (tuning_data.number_of_evaluated_valid_configurations >=
                self._fraction * tuning_data.constrained_search_space_size)

    def progress(self, tuning_data: TuningData):
        return (tuning_data.number_of_evaluated_valid_configurations
                / tuning_data.constrained_search_space_size
                / self._fraction)

    def to_json(self):
        return {'kind': 'ValidFraction', 'fraction': self._fraction}
