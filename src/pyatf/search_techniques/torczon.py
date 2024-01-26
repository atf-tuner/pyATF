import copy
import math
import random
from enum import Enum
from typing import Set, Dict, Optional

from pyatf.search_techniques.search_technique import SearchTechnique
from pyatf.tuning_data import Cost, Coordinates

INIT_SIMPLEX_NORMALIZED_SIDE_LENGTH = 0.1  # in (0,0.5]


def clamp_coordinates_capped(coordinates: Coordinates):
    return tuple(max(math.ulp(0.0), min(1.0, c)) for c in coordinates)


class Torczon(SearchTechnique):
    class State(Enum):
        TORC_INITIAL = 0
        TORC_REFLECTED = 1
        TORC_EXPANDED = 2

    class Simplex:
        vertices = None
        best_vertex = None

    def __init__(self):
        self._param_expansion: float = 2.0
        self._param_contraction: float = 0.5

        self._dimensionality: Optional[int] = None
        self._base_simplex: Optional[Torczon.Simplex] = None
        self._test_simplex: Optional[Torczon.Simplex] = None
        self._current_simplex: Optional[Torczon.Simplex] = None
        self._current_vertex: Optional[int] = None
        self._current_center: Optional[int] = None
        self._current_state: Optional[Torczon.State] = None
        self._best_cost: Optional[Cost] = None
        self._cost_improved: Optional[bool] = None

    def initialize(self, dimensionality: int):
        self._dimensionality = dimensionality
        self._base_simplex = Torczon.Simplex()
        self._base_simplex.vertices = self._initial_simplex_vertices()
        self._base_simplex.best_vertex = 0
        self._current_simplex = self._base_simplex
        self._test_simplex = Torczon.Simplex()
        self._current_state = Torczon.State.TORC_INITIAL
        self._current_vertex = 0
        self._current_center = 0
        self._cost_improved = True
        self._best_cost = float('inf')

    def finalize(self):
        pass

    def get_next_coordinates(self) -> Set[Coordinates]:
        if self._current_vertex == self._dimensionality + 1:
            self._generate_next_simplex()
        return {clamp_coordinates_capped(self._current_simplex.vertices[self._current_vertex])}

    def report_costs(self, costs: Dict[Coordinates, Cost]):
        if len(costs) != 1:
            raise ValueError('expecting costs for exactly one coordinate')
        coordinates, cost = next(iter(costs.items()))
        if cost is None:
            cost = float('inf')
        if cost < self._best_cost:
            self._best_cost = cost
            self._cost_improved = True
            self._current_simplex.best_vertex = self._current_vertex
            if self._current_state == Torczon.State.TORC_INITIAL:
                self._current_center = self._current_vertex
        self._current_vertex += 1

    def _initial_simplex_vertices(self):
        vertices = []
        base_vertex_coords = list(1.0 - random.random() for _ in range(self._dimensionality))
        vertices.append(tuple(base_vertex_coords))
        for i in range(self._dimensionality):
            v = base_vertex_coords
            if v[i] <= 0.5:
                v[i] += INIT_SIMPLEX_NORMALIZED_SIDE_LENGTH
            else:
                v[i] -= INIT_SIMPLEX_NORMALIZED_SIDE_LENGTH
            vertices.append(tuple(v))
        return vertices

    def _expand_base_simplex_vertices(self, factor: float = None):
        if factor is None:
            factor = self._param_expansion
        expanded_vertices = []
        center = self._base_simplex.vertices[self._current_center]
        for v in self._base_simplex.vertices:
            new_v = clamp_coordinates_capped(tuple(
                center[d] * (1 - factor) + v[d] * factor
                for d in range(self._dimensionality)
            ))
            expanded_vertices.append(new_v)
        return expanded_vertices

    def _reflect_base_simplex_vertices(self):
        return self._expand_base_simplex_vertices(-1)

    def _contract_base_simplex_vertices(self):
        return self._expand_base_simplex_vertices(self._param_contraction)

    def _switch_state(self, new_state: 'Torczon.State'):
        self._current_state = new_state
        self._current_vertex = 0
        self._cost_improved = False

    def _generate_next_simplex(self):
        match self._current_state:
            case Torczon.State.TORC_INITIAL:
                self._test_simplex.vertices = self._reflect_base_simplex_vertices()
                self._test_simplex.best_vertex = 0
                self._current_simplex = self._test_simplex
                self._switch_state(Torczon.State.TORC_REFLECTED)
            case Torczon.State.TORC_REFLECTED:
                if self._cost_improved:
                    self._base_simplex = copy.copy(self._test_simplex)
                    self._test_simplex.vertices = self._expand_base_simplex_vertices()
                    self._test_simplex.best_vertex = 0
                    self._current_simplex = self._test_simplex
                    self._switch_state(Torczon.State.TORC_EXPANDED)
                else:
                    self._base_simplex.vertices = self._contract_base_simplex_vertices()
                    self._base_simplex.best_vertex = 0
                    self._current_simplex = self._base_simplex
                    self._best_cost = float('inf')
                    self._current_center = 0
                    self._switch_state(Torczon.State.TORC_INITIAL)
            case Torczon.State.TORC_EXPANDED:
                if self._cost_improved:
                    self._base_simplex = copy.copy(self._test_simplex)
                self._current_center = self._base_simplex.best_vertex
                self._test_simplex.vertices = self._reflect_base_simplex_vertices()
                self._test_simplex.best_vertex = 0
                self._current_simplex = self._test_simplex
                self._switch_state(Torczon.State.TORC_REFLECTED)
