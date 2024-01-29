import math
import random
from enum import Enum
from typing import Set, Dict, Optional, List

from pyatf.search_techniques.search_technique import SearchTechnique
from pyatf.tuning_data import Cost, Coordinates


def interp(a: float, b: float, t: float):
    if t < 0 or t > 1:
        raise ValueError('expecting t to be in [0,1]')
    return a + t * (b - a)


def get_step_size(time: int, temp: float):
    return math.exp(-(20.0 + time / 100.0) / (temp + 1.0))


def AcceptanceFunction(e: float, e_new: float, temp: float):
    if e >= e_new:
        return 1.0
    if temp == 0:
        return 0.0
    if 50.0 * (e_new - e) / temp > 10:
        return 0.0
    return math.exp(50.0 * (e - e_new) / temp)


def relative(result1: float, result2: Optional[float]):
    if result2 == 0.0:
        return result1 * float('inf')
    return result1 / result2


def clamp_coordinates_capped(coordinates: Coordinates):
    return tuple(max(math.ulp(0.0), min(1.0, c)) for c in coordinates)


class SimulatedAnnealing(SearchTechnique):
    class State(Enum):
        INITIALIZATION = 0
        EXPLORE_PLUS = 1
        EXPLORE_MINUS = 2

    def __init__(self):
        # holds the default number of steps to interpolate
        self._default_interp_steps: int = 100
        # holds a list of temperatures to interpolate between
        self._temps: List[float] = [30.0, 0.0]

        # holds a set of numbers that determine how to interpolate between temps elements
        self._interp_steps: Optional[List[int]] = None
        # current state
        self._current_state: Optional[SimulatedAnnealing.State] = None
        # number of determined next configurations/time
        self._time: Optional[int] = None
        # maximum number of configurations within this cooling period/maximum cooling time
        self._max_time: Optional[int] = None
        # indicates the current parameter to mutate
        self._current_parameter: Optional[int] = None
        # dimensionality of coordinate space
        self._dimensionality: Optional[int] = None
        # holds the current best result found
        self._best_result: Optional[Cost] = None
        # holds the current temperature
        self._temp: Optional[float] = None
        # holds the current step size range
        self._step_size: Optional[float] = None
        # holds the current coordinates
        self._current_coordinates: Optional[Coordinates] = None
        # holds the best coordinates found yet
        self._best_coordinates: Optional[Coordinates] = None
        # the temperature schedule
        self._schedule: Optional[List[float]] = None
        # vector that holds all potentially next points
        self._neighbors: Optional[Dict[Coordinates, Cost]] = None

    def initialize(self, dimensionality: int):
        self._dimensionality = dimensionality
        self._current_state = SimulatedAnnealing.State.INITIALIZATION
        self._time = 0
        self._interp_steps = [self._default_interp_steps] * (len(self._temps) - 1)
        self._schedule = []
        for t in range(len(self._temps) - 1):
            for steps in range(self._interp_steps[t], 0, -1):
                self._schedule.append(interp(self._temps[t + 1], self._temps[t], steps / self._interp_steps[t]))
        self._schedule.append(self._temps[-1])
        self._max_time = len(self._schedule) - 1
        self._neighbors = {}

    def finalize(self):
        pass

    def get_next_coordinates(self) -> Set[Coordinates]:
        match self._current_state:
            case SimulatedAnnealing.State.INITIALIZATION:
                self._current_parameter = 0
                self._temp = self._schedule[min(self._time, self._max_time)]
                self._step_size = get_step_size(self._time, self._temp)
                self._current_coordinates = tuple(1.0 - random.random() for _ in range(self._dimensionality))
                self._neighbors[self._current_coordinates] = 0.0
                return {clamp_coordinates_capped(self._current_coordinates)}
            case SimulatedAnnealing.State.EXPLORE_PLUS:
                if self._current_coordinates[self._current_parameter] < 1.0:
                    new_coordinates = list(self._current_coordinates)
                    new_coordinates[self._current_parameter] += self._step_size * random.random()
                    new_coordinates = tuple(new_coordinates)
                    self._neighbors[new_coordinates] = 0.0
                    if self._current_coordinates[self._current_parameter] <= 0.0:
                        self._current_state = SimulatedAnnealing.State.EXPLORE_MINUS
                    return {clamp_coordinates_capped(new_coordinates)}
                else:
                    self._current_state = SimulatedAnnealing.State.EXPLORE_MINUS
                    new_coordinates = list(self._current_coordinates)
                    new_coordinates[self._current_parameter] -= self._step_size * random.random()
                    new_coordinates = tuple(new_coordinates)
                    self._neighbors[new_coordinates] = 0.0
                    return {clamp_coordinates_capped(new_coordinates)}
            case SimulatedAnnealing.State.EXPLORE_MINUS:
                new_coordinates = list(self._current_coordinates)
                new_coordinates[self._current_parameter] -= self._step_size * random.random()
                new_coordinates = tuple(new_coordinates)
                self._neighbors[new_coordinates] = 0.0
                return {clamp_coordinates_capped(new_coordinates)}

    def report_costs(self, costs: Dict[Coordinates, Cost]):
        if len(costs) != 1:
            raise ValueError('expecting costs for exactly one coordinate')
        coordinates, cost = next(iter(costs.items()))
        if cost is None:
            cost = float('inf')
        match self._current_state:
            case SimulatedAnnealing.State.INITIALIZATION:
                self._neighbors[coordinates] = cost
                self._best_coordinates = coordinates
                self._best_result = cost
                self._current_state = SimulatedAnnealing.State.EXPLORE_PLUS
            case SimulatedAnnealing.State.EXPLORE_PLUS:
                self._neighbors[coordinates] = cost
                if cost < self._best_result:
                    self._best_coordinates = coordinates
                    self._best_result = cost
                self._current_state = SimulatedAnnealing.State.EXPLORE_MINUS
            case SimulatedAnnealing.State.EXPLORE_MINUS:
                self._neighbors[coordinates] = cost
                if cost < self._best_result:
                    self._best_coordinates = coordinates
                    self._best_result = cost
                self._current_parameter += 1
                if self._current_parameter == self._dimensionality:
                    self._current_parameter = 0
                    current_result = None
                    while True:
                        if not self._neighbors:
                            self._current_coordinates = self._best_coordinates
                            current_result = self._best_result
                            break
                        candidate_coordinates, candidate_cost = random.choice(list(self._neighbors.items()))
                        if random.random() < AcceptanceFunction(1.0,
                                                                relative(candidate_cost, self._best_result),
                                                                self._temp):
                            self._current_coordinates = candidate_coordinates
                            current_result = candidate_cost
                            break
                        del self._neighbors[candidate_coordinates]
                    self._time += 1
                    if self._time > self._max_time:
                        self._time -= self._max_time
                    self._temp = self._schedule[min(self._time, self._max_time)]
                    self._step_size = get_step_size(self._time, self._temp)
                    self._neighbors.clear()
                    self._neighbors[self._current_coordinates] = current_result
                    self._current_state = SimulatedAnnealing.State.EXPLORE_PLUS
                else:
                    self._current_state = SimulatedAnnealing.State.EXPLORE_PLUS
