import os
import subprocess
import time
from typing import Optional, Any, Dict, Tuple

from pyatf.tuning_data import Configuration, CostFunctionError, Cost, MetaData


class CostFunction:
    def __init__(self, *run_command: str):
        self._run_command: Tuple[str] = tuple(run_command)
        self._compile_command: Optional[Tuple[str]] = None
        self._cost_file: Optional[str] = None

    def compile_command(self, *compile_command: str):
        self._compile_command = tuple(compile_command)
        return self

    def cost_file(self, cost_file: str):
        self._cost_file = cost_file
        return self

    def __call__(self, configuration: Configuration) -> Tuple[Cost, MetaData]:
        # add configuration to environment variables
        augmented_env = os.environ.copy()
        for tp_name, tp_value in configuration.items():
            augmented_env[tp_name] = str(tp_value)

        # collect meta-data for logging
        meta_data: Dict[str, Any] = {}

        # execute compile command
        if self._compile_command is not None:
            compile_start = time.perf_counter_ns()
            ret = subprocess.run(self._compile_command, env=augmented_env)
            compile_end = time.perf_counter_ns()
            meta_data['compile_command_exit_code'] = ret.returncode
            meta_data['compile_command_ns'] = compile_end - compile_start
            if meta_data['compile_command_exit_code'] != 0:
                raise CostFunctionError(meta_data) \
                    from RuntimeError('error while executing compile command: ' + ' '.join(self._compile_command))

        # execute run command
        run_start = time.perf_counter_ns()
        ret = subprocess.run(self._run_command, env=augmented_env)
        run_end = time.perf_counter_ns()
        meta_data['run_command_exit_code'] = ret.returncode
        meta_data['run_command_ns'] = run_end - run_start
        if meta_data['run_command_exit_code'] != 0:
            raise CostFunctionError(meta_data) \
                from RuntimeError('error while executing run command: ' + ' '.join(self._run_command))

        # get configuration cost
        if self._cost_file is None:
            return float(meta_data['run_command_ns']), meta_data
        else:
            with open(self._cost_file, 'r') as cost_file:
                return float(cost_file.read()), meta_data
