import math
import random
from typing import Set, Dict, Optional, List

from pyatf.search_techniques.search_technique import SearchTechnique
from pyatf.tuning_data import Cost, Coordinates

# number of vectors of the population with a minimum of 4
NUM_VECTORS = 30
# number of vectors to create the donor vector
NUM_MUT_VECTORS = 3
# mutation factor which scale the donor vector
F_VAL = 0.7  # 0.4 - 1.0
# Crossover-Rate to combine the donor vector with the current vector
CR = 0.2  # 0.0 - 1.0
# number of retries to calculate a new trial vector if the current one isn't in the search space
INVALID_RETRIES = 1


def clamp_coordinates_capped(coordinates: Coordinates):
    return tuple(max(math.ulp(0.0), min(1.0, c)) for c in coordinates)


class DifferentialEvolution(SearchTechnique):
    def __init__(self):
        self._dimensionality: Optional[int] = None
        # vector of points from population
        self._vector_population: Optional[List[Coordinates]] = None
        # trial vector
        self._trial_vector: Optional[List[float]] = None
        # vector of costs from each point of population
        self._population_costs: Optional[List[Cost]] = None
        # counter of the vectors of population
        self._current_vec: Optional[int] = None

    def initialize(self, dimensionality: int):
        self._dimensionality = dimensionality
        self._current_vec = 0
        self._population_init()
        self._trial_vector = list(1.0 - random.random() for _ in range(self._dimensionality))

    def finalize(self):
        pass

    def get_next_coordinates(self) -> Set[Coordinates]:
        if self._population_costs[self._current_vec] is None:
            return {clamp_coordinates_capped(self._vector_population[self._current_vec])}
        else:
            self._set_trial_vector()
            return {clamp_coordinates_capped(tuple(self._trial_vector))}

    def report_costs(self, costs: Dict[Coordinates, Cost]):
        if len(costs) != 1:
            raise ValueError('expecting costs for exactly one coordinate')
        coordinates, cost = next(iter(costs.items()))
        if cost is None:
            cost = float('inf')
        if self._population_costs[self._current_vec] is None:
            if cost == float('inf'):
                self._vector_population[self._current_vec] = tuple(1.0 - random.random()
                                                                   for _ in range(self._dimensionality))
            else:
                self._population_costs[self._current_vec] = cost
        elif cost <= self._population_costs[self._current_vec]:
            self._vector_population[self._current_vec] = tuple(self._trial_vector)
            self._population_costs[self._current_vec] = cost

        if self._current_vec < NUM_VECTORS - 1:
            self._current_vec += 1
        else:
            self._current_vec = 0

    def _population_init(self):
        self._vector_population = []
        self._population_costs = []
        for _ in range(NUM_VECTORS):
            self._vector_population.append(tuple(1.0 - random.random() for __ in range(self._dimensionality)))
            self._population_costs.append(None)

    def _random_vectors(self):
        vecs = [0] * NUM_MUT_VECTORS
        for i in range(NUM_MUT_VECTORS):
            vecs[i] = random.randint(0, NUM_VECTORS - 1)
            j = 0
            while j < i:
                if (vecs[i] == vecs[j] and i != j) or vecs[i] == self._current_vec:
                    vecs[i] = random.randint(0, NUM_VECTORS - 1)
                else:
                    j += 1
        return vecs

    def _set_trial_vector(self):
        for _ in range(INVALID_RETRIES):
            random_param = random.randint(0, self._dimensionality - 1)
            mutation_vec_indices = self._random_vectors()

            for i in range(self._dimensionality):
                if random.random() <= CR or i == random_param:
                    self._trial_vector[i] = self._get_donor_vector(i, mutation_vec_indices)
                else:
                    self._trial_vector[i] = self._vector_population[self._current_vec][i]

            if all(map(lambda c: 0.0 < c <= 1.0, self._trial_vector)):
                break
        for d in range(self._dimensionality):
            if self._trial_vector[d] <= 0.0 or self._trial_vector[d] > 1.0:
                self._trial_vector[d] = math.fmod(abs(self._trial_vector[d]), 1.0)

    def _get_donor_vector(self, param: int, mutation_vec_indices: List[int]):
        return (self._vector_population[mutation_vec_indices[0]][param]
                + F_VAL * (self._vector_population[mutation_vec_indices[1]][param] -
                           self._vector_population[mutation_vec_indices[2]][param]))
