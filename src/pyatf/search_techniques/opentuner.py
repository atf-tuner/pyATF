import math
from typing import Set, Dict, Optional

import argparse
import opentuner
from opentuner import ConfigurationManipulator, FloatParameter, Result
from opentuner.api import TuningRunManager
from opentuner.measurement.interface import DefaultMeasurementInterface

from pyatf.search_techniques.search_technique import SearchTechnique
from pyatf.tuning_data import Cost, Coordinates


class OpenTuner(SearchTechnique):
    def __init__(self, database_path: Optional[str] = None):
        self._dimensionality: Optional[int] = None
        parser = argparse.ArgumentParser(parents=opentuner.argparsers())
        self._args = parser.parse_args([])
        self._args.no_dups = True
        self._args.quiet = True
        if database_path is not None:
            self._args.database = database_path
        self._api = None
        self._desired_result = None
        self._coordinates = None

    def initialize(self, dimensionality: int):
        self._dimensionality = dimensionality
        manipulator = ConfigurationManipulator()
        for d in range(dimensionality):
            manipulator.add_parameter(FloatParameter(f'PARAM{d}', math.ulp(0.0), 1.0))
        interface = DefaultMeasurementInterface(args=self._args,
                                                manipulator=manipulator,
                                                project_name='atf',
                                                program_name='atf',
                                                program_version='1.0')
        self._api = TuningRunManager(interface, self._args)

    def finalize(self):
        self._api.finish()
        self._api = None
        self._dimensionality = None

    def get_next_coordinates(self) -> Set[Coordinates]:
        # get next desired result from OpenTuner
        self._desired_result = self._api.get_next_desired_result()
        while (self._desired_result is None
               or self._desired_result.configuration is None
               or self._desired_result.configuration.data is None):
            self._desired_result = self._api.get_next_desired_result()

        # convert to coordinates
        self._coordinates = tuple(
            self._desired_result.configuration.data[f'PARAM{d}']
            for d in range(self._dimensionality)
        )

        return {self._coordinates}

    def report_costs(self, costs: Dict[Coordinates, Cost]):
        if len(costs) != 1:
            raise ValueError('expecting costs for exactly one coordinate')
        coordinates, cost = next(iter(costs.items()))
        if coordinates != self._coordinates:
            raise ValueError(f'expecting cost for coordinates {self._coordinates}')
        if cost is not None:
            self._api.report_result(self._desired_result, Result(time=cost))
        else:
            self._api.report_result(self._desired_result, Result(state='ERROR', time=float('inf')))
