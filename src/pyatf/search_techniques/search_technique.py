from typing import Set, Dict

from pyatf.tuning_data import Cost, Coordinates


class SearchTechnique:
    def initialize(self, dimensionality: int):
        """
        Initializes the search technique.

        :param dimensionality: dimensionality "D" of the coordinate space
        """
        raise NotImplementedError

    def finalize(self):
        """
        Finalizes the search technique.
        """
        raise NotImplementedError

    def get_next_coordinates(self) -> Set[Coordinates]:
        """
        Returns the next coordinates in (0,1]^D for which the costs are requested.

        Function `get_next_coordinates()` is called by ATF before each call to `report_costs(...)`.

        :return: coordinates in (0,1]^D
        """
        raise NotImplementedError

    def report_costs(self, costs: Dict[Coordinates, Cost]):
        """
        Processes costs for coordinates requested via function `get_next_coordinates()`.

        Function `report_costs(...)` is called by ATF after each call to `get_next_coordinates()`.

        :param costs: coordinates mapped to their costs
        """
        raise NotImplementedError

    def to_json(self):
        """
        Returns the search technique in json format. Used for logging purposes only

        :return: The search technique in json format.
        """
        return {'kind': type(self).__name__}
