import argparse
import sys
from argparse import ArgumentTypeError, Namespace
from typing import Union, Callable, Any, Type, Iterable
from numbers import Number


def parse_cli_args():
    _geq_zero_i = make_half_interval_parser(0)
    _geq_zero_f = make_half_interval_parser(0.0)
    _01_open_ended_interval_f = make_interval_parser(0.0, 1.0, lower_closed=True, upper_closed=False)
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', '--batch_size', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-e', '--epochs', type=int, default=200, help='number of epochs to train for')
    parser.add_argument('-warm', type=int, default=-1, help='warm up training phase')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    # ================================================ QMargin Options ================================================
    kmargin_group = parser.add_argument_group("K-Margin Options")
    kmargin_group.add_argument('-k', '--k_warm', type=_geq_zero_i, default=1.0)
    kmargin_group.add_argument('-m', '--margin', type=_01_open_ended_interval_f, default=0.4)
    kmargin_action = parser.add_argument('--kmargin', action='store_true', default=False, help='use kmargin sampling')
    show_group_on_action(kmargin_group, kmargin_action)
    # ================================================ Optimizer Options ================================================
    sgd_group = set()
    adam_group = set()
    optimizer_choice = parser.add_argument('--optimizer', choices=('sgd', 'adam'), default='sgd',
                                           help='which optimizer to use')
    show_group_on_action(kmargin_group, kmargin_action)
    optimizer_parser = parser.add_argument_group("Optimizer Options")
    optimizer_parser.add_argument('-lr', type=_geq_zero_f, default=0.05, help='initial learning rate')
    arg = optimizer_parser.add_argument('--momentum', type=_geq_zero_f, default=0.0, help='momentum factor')
    sgd_group.add(arg)
    arg = optimizer_parser.add_argument('--weight_decay', type=_geq_zero_f, default=0.0,
                                        help='weight decay (L2 penalty)')
    sgd_group.add(arg)
    adam_group.add(arg)
    arg = optimizer_parser.add_argument('--dampening', type=_geq_zero_f, default=0.0, help='dampening for momentum')
    sgd_group.add(arg)
    arg = optimizer_parser.add_argument('--nesterov', type=bool, default=False, help='enables Nesterov momentum')
    sgd_group.add(arg)
    arg = optimizer_parser.add_argument('--betas', type=_geq_zero_f, default=(0.9, 0.999), nargs=2,
                                        help='coefficients used for computing running averages of gradient and its square')
    adam_group.add(arg)
    arg = optimizer_parser.add_argument('--eps', type=_geq_zero_f, default=1e-8,
                                        help='term added to the denominator to improve numerical stability')
    adam_group.add(arg)
    arg = optimizer_parser.add_argument('--amsgrad', type=bool, default=False,
                                        help='whether to use the AMSGrad variant of this algorithm from the paper `On the '
                                             'Convergence of Adam and Beyond`')
    adam_group.add(arg)
    show_group_on_value(adam_group, optimizer_choice, 'adam')
    show_group_on_value(sgd_group, optimizer_choice, 'sgd')
    return parser.parse_args()



def show_group_on_value(group: Union[Iterable[argparse.Action], argparse._ArgumentGroup],
                         action: argparse.Action, value):
    if isinstance(group, argparse._ArgumentGroup):
        group = group._group_actions

    args = list(s for s in sys.argv[1:] for s in s.split('='))
    if not any(arg in action.option_strings and args[i+1] == value for i, arg in enumerate(args)):
        for a in group:
            a.help = argparse.SUPPRESS

def show_group_on_action(group: Union[Iterable[argparse.Action], argparse._ArgumentGroup],
                         action: argparse.Action):
    if isinstance(group, argparse._ArgumentGroup):
        group = group._group_actions
    if not any(arg in action.option_strings for arg in sys.argv[1:]):
        for a in group:
            a.help = argparse.SUPPRESS

def show_on_action(element: argparse.Action, action: argparse.Action):
    if not any(arg in action.option_strings for arg in sys.argv[1:]):
        element.help = argparse.SUPPRESS

def show_on_any_action(element: argparse.Action, actions: Iterable[argparse.Action]):
    if not any(arg in action.option_strings for arg in sys.argv[1:] for action in actions):
        element.help = argparse.SUPPRESS

# https://izziswift.com/how-to-parse-multiple-nested-sub-commands-using-python-argparse/
def parse_args(parser, commands):
    # Divide argv by commands
    print(commands._group_actions)
    split_argv = [[]]
    for c in sys.argv[1:]:
        if c in commands._group_actions:
            print(c)
            split_argv.append([c])
        else:
            split_argv[-1].append(c)
    print(split_argv)
    # Initialize namespace
    args = Namespace()
    for c in commands._group_actions:
        setattr(args, c, None)

    print(args)
    # Parse each command
    parser.parse_args(split_argv[0], namespace=args)  # Without command
    for argv in split_argv[1:]:  # Commands
        cn = Namespace()
        setattr(args, argv[0], cn)
        commands._group_actions[argv[0]].parse_args(argv[1:], cn)
        # commands.choices[argv[0]].parse_args()
    return args


class _Interval:
    @property
    def value_type(self) -> Type:
        return self._value_type

    def __init__(self, lower: Number, upper: Number, lower_closed: bool = True, upper_closed: bool = False):
        self._lower = lower
        self._upper = upper
        self._lower_closed = lower_closed
        self._upper_closed = upper_closed
        self._in_lower = (lambda v: v >= self._lower) if lower_closed else (lambda v: v > self._lower)
        self._in_upper = (lambda v: v <= self._upper) if upper_closed else (lambda v: v < self._upper)

        if not isinstance(self._lower, type(self._upper)):
            if isinstance(self._lower, float) or isinstance(self._upper, float):
                self._value_type = float
            else:
                raise ValueError(f"Mismatching types between lower and upper: {type(lower)} != {type(upper)}")
        else:
            self._value_type = type(self._lower)

    def __str__(self):
        return f"{'[' if self._lower_closed else '<'}{self._lower}-{self._upper}{']' if self._upper_closed else '>'}"

    def contains(self, value: Number):
        return self._in_lower(value) and self._in_upper(value)


class _LowerHalfInterval:
    @property
    def value_type(self) -> Type:
        return self._value_type

    def __init__(self, lower: Number, inclusive: bool = True):
        self._lower = lower
        self._inclusive = inclusive
        self._in_lower = (lambda v: v >= self._lower) if inclusive else (lambda v: v > self._lower)
        self._value_type = type(self._lower)

    def __str__(self):
        return f"{self._lower}{'<=' if self._inclusive else '<'}"

    def contains(self, value: Number):
        return self._in_lower(value)


class _UpperHalfInterval:
    @property
    def value_type(self) -> Type:
        return self._value_type

    def __init__(self, upper: Number, inclusive: bool = True):
        self._upper = upper
        self._inclusive = inclusive
        self._in_upper = (lambda v: v <= self._upper) if inclusive else (lambda v: v < self._upper)
        self._value_type = type(self._upper)

    def __str__(self):
        return f"{self._upper}{'<=' if self._inclusive else '<'}"

    def contains(self, value: Number):
        return self._in_upper(value)


def make_half_interval_parser(limit: Number, upper: bool = False, inclusive: bool = True) -> Callable[[Any], Number]:
    half_interval = _UpperHalfInterval(limit, inclusive) if upper else _LowerHalfInterval(limit, inclusive)

    def half_interval_parser(value):
        value = half_interval.value_type(value)
        if not half_interval.contains(value):
            raise ArgumentTypeError(f'must be {half_interval}')
        return value

    half_interval_parser.__name__ = half_interval.value_type.__name__
    return half_interval_parser


def make_interval_parser(lower: Number, upper: Number, lower_closed: bool = True,
                         upper_closed: bool = False) -> Callable[[Any], Number]:
    interval = _Interval(lower, upper, lower_closed, upper_closed)

    def interval_parser(value):
        value = interval.value_type(value)
        if not interval.contains(value):
            raise ArgumentTypeError(f'must be within interval {interval}')
        return value
    interval_parser.__name__ = interval.value_type.__name__
    return interval_parser
