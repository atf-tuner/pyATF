import math
import random
from enum import Enum
from typing import Set, Dict, Optional, List

from pyatf.search_techniques.search_technique import SearchTechnique
from pyatf.tuning_data import Cost, Coordinates


def clamp_coordinates_capped(coordinates: Coordinates):
    return tuple(max(math.ulp(0.0), min(1.0, c)) for c in coordinates)


class PatternSearch(SearchTechnique):
    class State(Enum):
        INITIALIZATION = 0
        EXPLORATORY_PLUS = 1
        EXPLORATORY_MINUS = 2
        PATTERN = 3

    def __init__(self):
        # dimensionality of search space
        self._dimensionality: Optional[int] = None

        # base coordinates to go back to and its fitness
        self._base: Optional[Coordinates] = None
        self._base_fitness: Optional[Cost] = None
        # coordinates that moves through the search space and its fitness
        self._exploratory_coordinates: Optional[List[float]] = None
        self._exploratory_coordinates_fitness: Optional[Cost] = None
        # coordinates and its fitness after pattern move
        self._pattern_coordinates: Optional[List[float]] = None
        self._pattern_coordinates_fitness: Optional[Cost] = None
        # trigger to flag if Parameter has increased
        self._trigger: Optional[bool] = None
        # index of current parameter
        self._current_parameter: Optional[int] = None
        # current step size
        self._step_size: Optional[float] = None
        # current state
        self._current_state: Optional[PatternSearch.State] = None

    def initialize(self, dimensionality: int):
        self._dimensionality = dimensionality
        self._base = tuple(1.0 - random.random() for _ in range(self._dimensionality))
        self._trigger = False
        self._step_size = 0.1
        self._current_parameter = 0
        self._current_state = PatternSearch.State.INITIALIZATION

    def finalize(self):
        pass

    def get_next_coordinates(self) -> Set[Coordinates]:
        match self._current_state:
            case PatternSearch.State.INITIALIZATION:
                self._exploratory_coordinates = list(self._base)
                self._pattern_coordinates = list(self._base)
                return {clamp_coordinates_capped(self._base)}
            case PatternSearch.State.EXPLORATORY_PLUS:
                self._exploratory_coordinates[self._current_parameter] += self._step_size
                return {clamp_coordinates_capped(tuple(self._exploratory_coordinates))}
            case PatternSearch.State.EXPLORATORY_MINUS:
                if self._trigger:
                    self._exploratory_coordinates[self._current_parameter] -= 2 * self._step_size
                else:
                    self._exploratory_coordinates[self._current_parameter] -= self._step_size
                return {clamp_coordinates_capped(tuple(self._exploratory_coordinates))}
            case PatternSearch.State.PATTERN:
                return {clamp_coordinates_capped(tuple(self._pattern_coordinates))}

    def report_costs(self, costs: Dict[Coordinates, Cost]):
        if len(costs) != 1:
            raise ValueError('expecting costs for exactly one coordinate')
        coordinates, cost = next(iter(costs.items()))
        if cost is None:
            cost = float('inf')
        match self._current_state:
            case PatternSearch.State.INITIALIZATION:
                if cost == float('inf'):
                    self._base = tuple(1.0 - random.random() for _ in range(self._dimensionality))
                else:
                    self._base_fitness = cost
                    self._exploratory_coordinates_fitness = cost
                    self._pattern_coordinates_fitness = cost
                    self._current_state = PatternSearch.State.EXPLORATORY_PLUS
            case PatternSearch.State.EXPLORATORY_PLUS:
                if cost < self._exploratory_coordinates_fitness:
                    self._exploratory_coordinates[self._current_parameter] += self._step_size
                    if (self._exploratory_coordinates[self._current_parameter] <= 0.0
                            or self._exploratory_coordinates[self._current_parameter] > 1.0):
                        self._exploratory_coordinates[self._current_parameter] = math.fmod(
                            abs(self._exploratory_coordinates[self._current_parameter]), 1.0)
                    self._exploratory_coordinates_fitness = cost
                    self._trigger = True
                self._current_state = PatternSearch.State.EXPLORATORY_MINUS
            case PatternSearch.State.EXPLORATORY_MINUS:
                if cost < self._exploratory_coordinates_fitness:
                    if self._trigger:
                        self._exploratory_coordinates[self._current_parameter] -= 2 * self._step_size
                    else:
                        self._exploratory_coordinates[self._current_parameter] -= self._step_size
                    if (self._exploratory_coordinates[self._current_parameter] <= 0.0
                            or self._exploratory_coordinates[self._current_parameter] > 1.0):
                        self._exploratory_coordinates[self._current_parameter] = math.fmod(
                            abs(self._exploratory_coordinates[self._current_parameter]), 1.0)
                    self._exploratory_coordinates_fitness = cost
                self._trigger = False
                self._current_parameter += 1

                if self._current_parameter == self._dimensionality:
                    if self._exploratory_coordinates_fitness < self._pattern_coordinates_fitness:
                        for d in range(self._dimensionality):
                            self._pattern_coordinates[d] = 2 * self._exploratory_coordinates[d] - self._base[d]
                            if self._pattern_coordinates[d] <= 0.0 or self._pattern_coordinates[d] > 1.0:
                                self._pattern_coordinates[d] = math.fmod(abs(self._pattern_coordinates[d]), 1.0)
                        self._base = tuple(self._exploratory_coordinates)
                        self._base_fitness = self._exploratory_coordinates_fitness
                        self._exploratory_coordinates = self._pattern_coordinates.copy()
                        self._current_state = PatternSearch.State.PATTERN
                    else:
                        self._exploratory_coordinates = list(self._base)
                        self._exploratory_coordinates_fitness = self._base_fitness
                        self._pattern_coordinates = list(self._base)
                        self._pattern_coordinates_fitness = self._base_fitness
                        self._step_size *= 0.5
                        self._current_state = PatternSearch.State.EXPLORATORY_PLUS
                    self._current_parameter = 0
                else:
                    self._current_state = PatternSearch.State.EXPLORATORY_PLUS
            case PatternSearch.State.PATTERN:
                self._pattern_coordinates_fitness = cost
                self._exploratory_coordinates_fitness = cost
                self._current_state = PatternSearch.State.EXPLORATORY_PLUS
