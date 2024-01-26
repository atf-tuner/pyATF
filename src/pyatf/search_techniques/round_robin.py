from typing import Set, Dict, Iterable, Optional

from pyatf.search_techniques import SimulatedAnnealing, PatternSearch, Torczon, DifferentialEvolution
from pyatf.search_techniques.search_technique import SearchTechnique
from pyatf.tuning_data import Coordinates, Cost


class RoundRobin(SearchTechnique):
    def __init__(self, techniques: Iterable[SearchTechnique] = (
            SimulatedAnnealing(), DifferentialEvolution(), PatternSearch(), Torczon()
    )):
        self._techniques = tuple(techniques)
        self._num_techniques = len(self._techniques)
        self._current_technique_idx: Optional[int] = None

    def initialize(self, dimensionality: int):
        for technique in self._techniques:
            technique.initialize(dimensionality)
        self._current_technique_idx = 0

    def finalize(self):
        for technique in self._techniques:
            technique.finalize()

    def get_next_coordinates(self) -> Set[Coordinates]:
        return self._techniques[self._current_technique_idx].get_next_coordinates()

    def report_costs(self, costs: Dict[Coordinates, Cost]):
        self._techniques[self._current_technique_idx].report_costs(costs)
        self._current_technique_idx = (self._current_technique_idx + 1) % self._num_techniques
