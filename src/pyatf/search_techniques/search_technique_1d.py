from typing import Set, Dict

from pyatf.tuning_data import Cost, Index


class SearchTechnique1D:
    def initialize(self, search_space_size: int):
        """
        Initializes the search technique.

        :param search_space_size: the total number of configurations in the search space
        """
        raise NotImplementedError

    def finalize(self):
        """
        Finalizes the search technique.
        """
        raise NotImplementedError

    def get_next_indices(self) -> Set[Index]:
        """
        Returns the next indices in { 0 , ... , |SP|-1 } for which the costs are requested.

        Function `get_next_indices()` is called by ATF before each call to `report_costs(...)`.

        :return: indices in { 0 , ... , |SP|-1 }
        """
        raise NotImplementedError

    def report_costs(self, costs: Dict[Index, Cost]):
        """
        Processes costs for indices requested via function `get_next_indices()`.

        Function `report_costs(...)` is called by ATF after each call to `get_next_indices()`.

        :param costs: indices mapped to their costs
        """
        raise NotImplementedError

    def to_json(self):
        """
        Returns the search technique in json format. Used for logging purposes only

        :return: The search technique in json format.
        """
        return {'kind': type(self).__name__}
