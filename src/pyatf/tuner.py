import json
import signal
import time
from datetime import datetime
from math import floor, ceil
from pathlib import Path
from typing import Optional, Tuple, Union, TextIO, Set, Dict

from pyatf.abort_conditions import Evaluations
from pyatf.abort_conditions.abort_condition import AbortCondition
from pyatf.search_space import SearchSpace
from pyatf.search_techniques import AUCBandit
from pyatf.search_techniques.search_technique import SearchTechnique
from pyatf.search_techniques.search_technique_1d import SearchTechnique1D
from pyatf.tp import TP
from pyatf.tuning_data import Cost, TuningData, CostFunction, Coordinates, Index, CostFunctionError

# register SIGINT handler to gracefully terminate tuning run early
SIGINT_handlers = set()


class SIGINTHandler:
    def __init__(self):
        self.SIGINT_received = False

    def __enter__(self):
        SIGINT_handlers.add(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        SIGINT_handlers.remove(self)


original_sigint_handler = signal.getsignal(signal.SIGINT)


def sigint_handler(*args):
    global SIGINT_handlers
    if SIGINT_handlers:
        print('SIGINT received, terminating early')
        for handler in SIGINT_handlers:
            handler.SIGINT_received = True
    else:
        original_sigint_handler(*args)


signal.signal(signal.SIGINT, sigint_handler)


class Tuner:
    class TuningRun:
        def __init__(self,
                     tps: Tuple[TP, ...],
                     cost_function: CostFunction,
                     search_technique: Optional[Union[SearchTechnique, SearchTechnique1D]],
                     silent: Optional[bool],
                     log_file: Optional[str],
                     abort_condition: Optional[AbortCondition]):
            if tps is None:
                raise ValueError('missing call to `Tuner.tuning_parameters(...)`: no tuning parameters defined')

            # prepare search technique
            self._search_technique: SearchTechnique = search_technique
            if self._search_technique is None:
                self._search_technique = AUCBandit()
            if isinstance(self._search_technique, SearchTechnique):
                self._get_next_coordinates_or_indices = self._search_technique.get_next_coordinates
                self._coordinates_or_index_param_name = 'search_space_coordinates'
            else:
                self._get_next_coordinates_or_indices = self._search_technique.get_next_indices
                self._coordinates_or_index_param_name = 'search_space_index'
            self._coordinates_or_indices: Set[Union[Coordinates, Index]] = set()
            self._costs: Dict[Coordinates, Union[Coordinates, Index]] = {}

            # generate search space
            search_space_generation_start = time.perf_counter_ns()
            self._search_space = SearchSpace(*tps,
                                             enable_1d_access=isinstance(self._search_technique, SearchTechnique1D),
                                             silent=silent)
            search_space_generation_end = time.perf_counter_ns()
            self._search_space_generation_ns = search_space_generation_end - search_space_generation_start

            # prepare abort condition
            self._abort_condition: Optional[AbortCondition] = abort_condition
            if self._abort_condition is None:
                self._abort_condition = Evaluations(len(self._search_space))

            # tuning data
            self._tps: Tuple[TP, ...] = tps
            self._tuning_data: Optional[TuningData] = None
            self._cost_function: CostFunction = cost_function

            # progress data
            self._silent = silent
            self._log_file: Optional[TextIO] = None
            if log_file:
                Path(log_file).parent.mkdir(parents=True, exist_ok=True)
                self._log_file = open(log_file, 'w')
            self._last_log_dump: Optional[int] = None
            self._last_line_length: Optional[int] = None
            self._tuning_start_ns: Optional[int] = None

        def __del__(self):
            if self._log_file:
                self._log_file.close()

        @property
        def cost_function(self):
            return self._cost_function

        @property
        def abort_condition(self):
            return self._abort_condition

        @property
        def tuning_data(self):
            return self._tuning_data

        def _print_progress(self, timestamp: datetime, cost: Optional[Cost] = None):
            now = time.perf_counter_ns()
            elapsed_ns = now - self._tuning_start_ns
            elapsed_seconds = elapsed_ns // 1000000000
            elapsed_time_str = (f'{elapsed_seconds // 3600}'
                                f':{elapsed_seconds // 60 % 60:02d}'
                                f':{elapsed_seconds % 60:02d}')
            progress = self._abort_condition.progress(self._tuning_data)
            line = (f'\r{timestamp.strftime("%Y-%m-%dT%H:%M:%S")}'
                    f'    evaluations: {self._tuning_data.number_of_evaluated_configurations}'
                    f' (valid: {self._tuning_data.number_of_evaluated_valid_configurations})'
                    f', min. cost: {self._tuning_data.min_cost()}'
                    f', valid: {cost is not None}'
                    f', cost: {cost}')
            line_length = len(line)
            if line_length < self._last_line_length:
                line += ' ' * (self._last_line_length - line_length)
            print(line)
            if progress is None:
                spinner_char = ('-', '\\', '|', '/')[(elapsed_ns // 500000000) % 4]
                line = f'\rTuning: {spinner_char} {elapsed_time_str}\r'
                print(line, end='')
            else:
                if now > self._tuning_start_ns and progress > 0:
                    eta_seconds = ceil(((now - self._tuning_start_ns) / progress
                                        * (1 - progress)) / 1000000000)
                    eta_str = (f'{eta_seconds // 3600}'
                               f':{eta_seconds // 60 % 60:02d}'
                               f':{eta_seconds % 60:02d}')
                else:
                    eta_str = '?'
                filled = '█' * floor(progress * 80)
                empty = ' ' * ceil((1 - progress) * 80)
                line = (f'\rexploring search space: |{filled}{empty}|'
                        f' {progress * 100:6.2f}% {elapsed_time_str} (ETA: {eta_str})')
                print(line, end='')
            self._last_line_length = len(line)

        def initialize(self):
            # reset progress data
            self._tuning_start_ns = time.perf_counter_ns()
            self._last_line_length = 0

            # create tuning data
            self._tuning_data = TuningData(list(tp.to_json() for tp in self._tps),
                                           self._search_space.constrained_size,
                                           self._search_space.unconstrained_size,
                                           self._search_space_generation_ns,
                                           self._search_technique.to_json(),
                                           self._abort_condition.to_json())

            # write tuning data
            if self._log_file:
                json.dump(self._tuning_data.to_json(), self._log_file, indent=4)

            # initialize search technique
            self._search_technique.initialize(self._search_space.num_tps)

        def make_step(self):
            # get new coordinates
            if not self._coordinates_or_indices:
                if self._costs:
                    self._search_technique.report_costs(self._costs)
                    self._costs.clear()
                self._coordinates_or_indices.update(self._get_next_coordinates_or_indices())

            # get configuration
            coords_or_index = self._coordinates_or_indices.pop()
            config = self._search_space.get_configuration(coords_or_index)

            # run cost function
            valid = True
            meta_data = None
            cost = None
            try:
                cost_function_return_values = self._cost_function(config)
                if isinstance(cost_function_return_values, tuple):
                    cost, meta_data = cost_function_return_values
                else:
                    cost = cost_function_return_values
            except CostFunctionError as e:
                meta_data = e.meta_data
                valid = False
            timestamp = self._tuning_data.record_evaluation(config, valid, cost, meta_data, **{
                self._coordinates_or_index_param_name: coords_or_index
            })
            self._costs[coords_or_index] = cost

            # print progress and dump log file (at most once every 5 minutes)
            if not self._silent:
                self._print_progress(timestamp, cost)
            if self._log_file and (self._last_log_dump is None or time.perf_counter_ns() - self._last_log_dump > 3e11):
                json.dump(self._tuning_data.to_json(), self._log_file, indent=4)
                self._last_log_dump = time.perf_counter_ns()

        def finalize(self, sigint_received: bool = False):
            self._search_technique.finalize()
            self._tuning_data.record_tuning_finished(sigint_received)

            # write tuning data to file
            if self._log_file:
                self._log_file.seek(0)
                json.dump(self._tuning_data.to_json(), self._log_file, indent=4)
                self._log_file.truncate()
                self._log_file.close()
                self._log_file = None

            if not self._silent:
                print('\nfinished tuning')
                if self._tuning_data.min_cost() is not None:
                    print('best configuration:')
                    for tp_name, tp_value in self._tuning_data.configuration_of_min_cost().items():
                        print(f'    {tp_name} = {tp_value}')
                    print(f'min cost: {self._tuning_data.min_cost()}')

    def __init__(self):
        self._tps: Optional[Tuple[TP, ...]] = None
        self._search_technique: Optional[Union[SearchTechnique, SearchTechnique1D]] = None
        self._silent = False
        self._log_file: Optional[str] = None

        self._tuning_run: Optional[Tuner.TuningRun] = None

    def tuning_parameters(self, *tps: TP):
        if self._tuning_run is not None:
            raise ValueError('cannot change tuning parameters while tuning')
        self._tps = tuple(tps)
        return self

    def search_technique(self, search_technique: Union[SearchTechnique, SearchTechnique1D]):
        if self._tuning_run is not None:
            raise ValueError('cannot change search technique while tuning')
        self._search_technique = search_technique
        return self

    def silent(self, silent: bool):
        if self._tuning_run is not None:
            raise ValueError('cannot change silent property while tuning')
        self._silent = silent
        return self

    def log_file(self, log_file: str):
        if self._tuning_run is not None:
            raise ValueError('cannot change log file while tuning')
        self._log_file = log_file
        return self

    def make_step(self, cost_function: CostFunction, progress: float = None):
        if self._tuning_run is None:
            # create & initialize tuning run
            self._tuning_run = Tuner.TuningRun(
                self._tps,
                cost_function,
                self._search_technique,
                self._silent,
                self._log_file,
                None
            )
            self._tuning_run.initialize()
        if cost_function != self._tuning_run.cost_function:
            raise ValueError('a tuning run with another cost function is already in progress')
        self._tuning_run.make_step()

    def get_tuning_data(self):
        if self._tuning_run is None:
            raise ValueError('cannot get tuning data, because no tuning run is in progress')
        return self._tuning_run.tuning_data

    def tune(self, cost_function: CostFunction, abort_condition: Optional[AbortCondition] = None):
        if self._tuning_run is not None:
            raise ValueError('cannot start tuning while another tuning run is still in progress')

        # create tuning run
        self._tuning_run = Tuner.TuningRun(
            self._tps,
            cost_function,
            self._search_technique,
            self._silent,
            self._log_file,
            abort_condition
        )

        # run tuning loop
        self._tuning_run.initialize()
        with SIGINTHandler() as h:
            while not self._tuning_run.abort_condition.stop(self._tuning_run.tuning_data) and not h.SIGINT_received:
                self._tuning_run.make_step()
            self._tuning_run.finalize(h.SIGINT_received)

        # return tuning result
        config = self._tuning_run.tuning_data.configuration_of_min_cost()
        min_cost = self._tuning_run.tuning_data.min_cost()
        tuning_data = self._tuning_run.tuning_data
        self._tuning_run = None
        return config, min_cost, tuning_data
