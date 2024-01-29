from pyatf.abort_conditions.abort_condition import AbortCondition
from pyatf.tuning_data import TuningData


class Evaluations(AbortCondition):
    def __init__(self, evaluations: int):
        self._evaluations = evaluations

    def stop(self, tuning_data: TuningData):
        return tuning_data.number_of_evaluated_configurations >= self._evaluations

    def progress(self, tuning_data: TuningData):
        return min(1.0, tuning_data.number_of_evaluated_configurations / self._evaluations)

    def to_json(self):
        return {'kind': 'Evaluations', 'evaluations': self._evaluations}


class ValidEvaluations(AbortCondition):
    def __init__(self, valid_evaluations: int):
        self._valid_evaluations = valid_evaluations

    def stop(self, tuning_data: TuningData):
        return tuning_data.number_of_evaluated_valid_configurations >= self._valid_evaluations

    def progress(self, tuning_data: TuningData):
        return min(1.0, tuning_data.number_of_evaluated_valid_configurations / self._valid_evaluations)

    def to_json(self):
        return {'kind': 'ValidEvaluations', 'valid_evaluations': self._valid_evaluations}
