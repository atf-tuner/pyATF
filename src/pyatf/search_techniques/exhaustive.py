from typing import Set, Dict, Optional

from pyatf.search_techniques.search_technique_1d import SearchTechnique1D
from pyatf.tuning_data import Cost, Index


class Exhaustive(SearchTechnique1D):
    def __init__(self):
        self._next_index: int = 0
        self._search_space_size: Optional[int] = None

    def initialize(self, search_space_size: int):
        self._search_space_size = search_space_size

    def finalize(self):
        pass

    def get_next_indices(self) -> Set[Index]:
        indices = {self._next_index}
        self._next_index += 1
        if self._next_index >= self._search_space_size:
            self._next_index = 0
        return indices

    def report_costs(self, costs: Dict[Index, Cost]):
        pass
