import inspect
from typing import Callable, Optional

from pyatf.range import Range


class TP:
    def __init__(self, name: str, values: Range, constraint: Optional[Callable[..., bool]] = None):
        self._name = name
        self._values = values
        self._constraint = constraint

    def __repr__(self):
        return self._name

    @property
    def name(self):
        return self._name

    @property
    def values(self):
        return self._values

    @property
    def constraint(self):
        return self._constraint

    def to_json(self):
        json = {
            'name': self._name,
            'range': self._values.to_json()
        }
        if self._constraint is not None:
            json['constraint'] = inspect.getsource(self._constraint)
        return json
