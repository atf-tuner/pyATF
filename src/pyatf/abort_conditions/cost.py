from pyatf.abort_conditions.abort_condition import AbortCondition
from pyatf.tuning_data import TuningData, Cost as TuningDataCost


class Cost(AbortCondition):
    def __init__(self, cost: TuningDataCost):
        self._cost = cost

    def stop(self, tuning_data: TuningData):
        best_result = tuning_data.min_cost()
        return best_result is not None and best_result <= self._cost

    def to_json(self):
        return {'kind': 'Cost', 'cost': self._cost}
