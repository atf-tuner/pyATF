import time
from typing import Type, Tuple, Any, Optional, Dict

from pyatf.tuning_data import Configuration, Cost


class CostFunction:
    class CostFunctionWithTypeAndArgs:
        def __init__(self, type_to_tune: Type, *call_args, **call_kwargs):
            self._type_to_tune = type_to_tune
            self._call_args: Tuple[Any, ...] = call_args
            self._call_kwargs: Dict[str, Any] = call_kwargs

        def __call__(self, configuration: Configuration) -> Cost:
            # create new instance of type to tune for current configuration
            instance = self._type_to_tune(configuration)

            # measure runtime of calling instance
            start = time.perf_counter_ns()
            instance(*self._call_args, **self._call_kwargs)
            end = time.perf_counter_ns()

            return end - start

    def __init__(self, type_to_tune: Type):
        self._type_to_tune = type_to_tune

    def __call__(self, *args, **kwargs):
        return CostFunction.CostFunctionWithTypeAndArgs(self._type_to_tune, *args, **kwargs)
