import math
import operator
import random
from typing import Set, Dict, Iterable, Optional, List

from pyatf.search_techniques import SimulatedAnnealing, PatternSearch, Torczon
from pyatf.search_techniques.search_technique import SearchTechnique
from pyatf.tuning_data import Coordinates, Cost

DEFAULT_C = 0.05
DEFAULT_WINDOW_SIZE = 500


class AUCBandit(SearchTechnique):
    class HistoryEntry:
        technique_idx: int = None
        cost_has_improved: bool = None

    def __init__(self, techniques: Iterable[SearchTechnique] = (SimulatedAnnealing(), PatternSearch(), Torczon()),
                 c: float = DEFAULT_C, window_size: int = DEFAULT_WINDOW_SIZE):
        self._techniques = tuple(techniques)
        self._num_techniques = len(self._techniques)
        self._c = c
        self._window_size = window_size

        self._current_technique_idx: Optional[int] = None
        self._current_best_cost: Optional[Cost] = None
        self._history: Optional[List[AUCBandit.HistoryEntry]] = None
        self._uses: Optional[List[int]] = None
        self._raw_auc: Optional[List[int]] = None
        self._decay: Optional[List[int]] = None

    def initialize(self, dimensionality: int):
        for technique in self._techniques:
            technique.initialize(dimensionality)
        self._current_technique_idx = 0
        self._current_best_cost = float('inf')
        self._history = []
        self._uses = [0] * self._num_techniques
        self._raw_auc = [0] * self._num_techniques
        self._decay = [0] * self._num_techniques

    def finalize(self):
        for technique in self._techniques:
            technique.finalize()

    def get_next_coordinates(self) -> Set[Coordinates]:
        self._current_technique_idx = self._get_best_technique_idx()
        return self._techniques[self._current_technique_idx].get_next_coordinates()

    def report_costs(self, costs: Dict[Coordinates, Cost]):
        self._techniques[self._current_technique_idx].report_costs(costs)

        min_cost = min(costs.values())
        cost_has_improved = min_cost is not None and min_cost < self._current_best_cost
        if cost_has_improved:
            self._current_best_cost = min_cost

        self._add_to_history(self._current_technique_idx, cost_has_improved)

    def _add_to_history(self, technique_idx: int, cost_has_improved: bool):
        if len(self._history) == self._window_size:
            oldest_technique = self._history[0]
            self._uses[oldest_technique.technique_idx] -= 1
            self._raw_auc[oldest_technique.technique_idx] -= self._decay[oldest_technique.technique_idx]
            if oldest_technique.cost_has_improved:
                self._decay[oldest_technique.technique_idx] -= 1
            del self._history[0]

        self._uses[technique_idx] += 1
        if cost_has_improved:
            self._raw_auc[technique_idx] += self._uses[technique_idx]
            self._decay[technique_idx] += 1
        self._history.append(AUCBandit.HistoryEntry())
        self._history[-1].technique_idx = technique_idx
        self._history[-1].cost_has_improved = cost_has_improved

    def _calculate_auc(self, technique_idx: int):
        if self._uses[technique_idx] > 0:
            return self._raw_auc[technique_idx] * 2.0 / (self._uses[technique_idx] * (self._uses[technique_idx] + 1.0))
        else:
            return 0.0

    def _calculate_exploration_value(self, technique_idx: int):
        if self._uses[technique_idx] > 0:
            return math.sqrt(2.0 * math.log2(len(self._history)) / self._uses[technique_idx])
        else:
            return float('inf')

    def _calculate_score(self, technique_idx: int):
        return self._calculate_auc(technique_idx) + self._c * self._calculate_exploration_value(technique_idx)

    def _get_best_technique_idx(self):
        indices = list(range(self._num_techniques))
        random.shuffle(indices)
        technique_idx_with_max_score, _ = max(enumerate(map(lambda idx: (idx, self._calculate_score(idx)), indices)),
                                              key=operator.itemgetter(1))
        return technique_idx_with_max_score
