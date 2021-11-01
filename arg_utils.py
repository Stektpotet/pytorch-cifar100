from argparse import ArgumentTypeError
from typing import Union, Callable, Any
from numbers import Number


class _Interval:
    def __init__(self, lower: Number, upper: Number, lower_closed: bool = True, upper_closed: bool = False):
        self._lower = lower
        self._upper = upper
        self._lower_closed = lower_closed
        self._upper_closed = upper_closed
        self._in_lower = (lambda v: v >= self._lower) if lower_closed else (lambda v: v > self._lower)
        self._in_upper = (lambda v: v <= self._upper) if upper_closed else (lambda v: v < self._upper)

    def __str__(self):
        return f"{'[' if self._lower_closed else '<'}{self._lower}-{self._upper}{']' if self._upper_closed else '>'}"

    def contains(self, value: Number):
        return self._in_lower(value) and self._in_upper(value)


def to_num(s) -> Union[int, float]:
    try:
        return int(s)
    except ValueError:
        return float(s)


def make_interval_parser(lower: Number, upper: Number, lower_closed: bool = True,
                         upper_closed: bool = False) -> Callable[[Any], Number]:
    interval = _Interval(lower, upper, lower_closed, upper_closed)

    def interval_parser(value):
        value = to_num(value)
        if not interval.contains(value):
            raise ArgumentTypeError(f'must be within interval {interval}')
        return value

    return interval_parser
