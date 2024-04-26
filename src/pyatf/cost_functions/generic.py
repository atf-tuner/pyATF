import os
import subprocess
import time
from typing import Optional

from pyatf.tuning_data import Configuration, CostFunctionError, Cost


class CostFunction:
    def __init__(self, run_command: str):
        self._run_command: str = run_command
        self._compile_command: Optional[str] = None
        self._cost_file: Optional[str] = None

    def compile_command(self, compile_command: str):
        self._compile_command = compile_command
        return self

    def cost_file(self, cost_file: str):
        self._cost_file = cost_file
        return self

    def __call__(self, configuration: Configuration) -> Cost:
        # add configuration to environment variables
        augmented_env = os.environ.copy()
        for tp_name, tp_value in configuration.items():
            augmented_env[tp_name] = str(tp_value)

        # execute compile command
        if self._compile_command is not None:
            ret = subprocess.run(self._compile_command, env=augmented_env, shell=True)
            if ret.returncode != 0:
                raise CostFunctionError('error while executing compile command: ' + self._compile_command)

        # execute run command
        run_start = time.perf_counter_ns()
        ret = subprocess.run(self._run_command, env=augmented_env, shell=True)
        run_end = time.perf_counter_ns()
        if ret.returncode != 0:
            raise CostFunctionError('error while executing run command: ' + self._run_command)

        # get configuration cost
        if self._cost_file is None:
            return float(run_end - run_start)
        else:
            with open(self._cost_file, 'r') as cost_file:
                return float(cost_file.read())
