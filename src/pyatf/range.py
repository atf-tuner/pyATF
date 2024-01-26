import inspect
import math
from typing import Optional, Callable, Union


class Range:
    def __len__(self):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def __getitem__(self, item: int):
        raise NotImplementedError

    def to_json(self):
        raise NotImplementedError


class Interval(Range):
    def __init__(self, start: Union[int, float], end: Union[int, float], step: Union[int, float] = 1,
                 generator: Optional[Callable[[Union[int, float]], Union[int, float]]] = None):
        arg_types = type(start), type(end), type(step)
        if any(map(lambda t: t not in (int, float), arg_types)):
            raise TypeError('invalid argument type: '
                            'expecting arguments start, end, and step to be of type int or float')
        if step == 0:
            raise ValueError('invalid argument value: expecting step != 0')

        if any(map(lambda t: t == float, arg_types)):
            self._num_values = 0
            if step < 0:
                while start + self._num_values * step >= end:
                    self._num_values += 1
            else:
                while start + self._num_values * step <= end:
                    self._num_values += 1
            self._start = 0
            self._end = self._num_values - 1
            self._step = 1
            if generator is None:
                self._generator = (
                    lambda i, float_start=start, float_end=end, float_step=step, num_values=self._num_values:
                    float_start + i * float_step
                )
            else:
                self._generator = (
                    lambda i, float_start=start, float_end=end, float_step=step, float_generator=generator,
                           num_values=self._num_values: float_generator(float_start + i * float_step)
                )
        else:
            self._start = start
            self._end = end
            self._step = step
            self._generator = generator
            if start == end:
                self._num_values = 1
            elif (step < 0 and start < end) or (step > 0 and start > end):
                self._num_values = 0
            else:
                self._num_values = math.ceil(abs((abs(end - start) + 1) / step))

        self._yield_func = None
        if self._step < 0:
            self._yield_func = self._dec_yield if self._generator is None else self._dec_generator_yield
        else:
            self._yield_func = self._inc_yield if self._generator is None else self._inc_generator_yield

    def _dec_yield(self):
        next_value = self._start
        while next_value >= self._end:
            yield next_value
            next_value += self._step

    def _inc_yield(self):
        next_value = self._start
        while next_value <= self._end:
            yield next_value
            next_value += self._step

    def _dec_generator_yield(self):
        next_value = self._start
        while next_value >= self._end:
            yield self._generator(next_value)
            next_value += self._step

    def _inc_generator_yield(self):
        next_value = self._start
        while next_value <= self._end:
            yield self._generator(next_value)
            next_value += self._step

    def __len__(self):
        return self._num_values

    def __iter__(self):
        yield from self._yield_func()

    def __getitem__(self, item: int):
        if item < 0 or item >= self._num_values:
            raise ValueError(f'out of bounds: {item}')
        value = self._start + item * self._step
        if self._generator is not None:
            value = self._generator(value)
        return value

    def to_json(self):
        json = {
            'kind': 'Interval',
            'start': self._start,
            'end': self._end,
            'step': self._step
        }
        if self._generator is not None:
            json['generator'] = inspect.getsource(self._generator)
        return json


class Set(Range):
    def __init__(self, *values):
        self._values = tuple(values)
        self._num_values = len(values)

    def __len__(self):
        return self._num_values

    def __iter__(self):
        yield from self._values

    def __getitem__(self, item):
        return self._values[item]

    def to_json(self):
        return {
            'kind': 'Set',
            'values': list(self._values)
        }
