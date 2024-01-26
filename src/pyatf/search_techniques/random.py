import random
from typing import Set, Dict, Optional

from pyatf.search_techniques.search_technique import SearchTechnique
from pyatf.tuning_data import Cost, Coordinates


class Random(SearchTechnique):
    def __init__(self):
        self._dimensionality: Optional[int] = None

    def initialize(self, dimensionality: int):
        self._dimensionality = dimensionality

    def finalize(self):
        pass

    def get_next_coordinates(self) -> Set[Coordinates]:
        return {tuple(1.0 - random.random() for _ in range(self._dimensionality))}

    def report_costs(self, costs: Dict[Coordinates, Cost]):
        pass
