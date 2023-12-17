"""
usage: python -m tests.benchmarks.multi_benchmark [-h] -m MODULE \
            [-s START] [-e END] [-n STEP] [-t TIMEOUT]

This module runs tests for a list of benchmark modules with a range of options. The benchmark
modules to test must have a `main` function that takes an input value as an option and a boolean
value that specifies whether the benchmark module itself should produce an output or not.

options:
  -h, --help            show this help message and exit
  -m MODULE, --module MODULE
                        The module(s) to test. Can be added multiple times.
  -s START, --start START
                        Start value for the range of options. Default is 5.
  -e END, --end END     End value for the range of options. Default is 100.
  -n STEP, --step STEP  Step value for the range of options. Default is 5.
  -t TIMEOUT, --timeout TIMEOUT
                        The maximum time allowed for a test to run. Default is 60

Example: python -m tests.benchmarks.multi_benchmark -m tests.benchmarks.benchmark_labyrinth_clsp \
    -m tests.benchmarks.benchmark_labyrinth -s 5 -e 15 -n 5
"""

from collections.abc import Callable, Iterable, Sequence
from typing import Any, Optional
from argparse import ArgumentParser
from importlib import import_module


def run(
    functions: Sequence[Callable[[int, bool], float]],
    n: int,
    skip: list[Callable[[int, bool], float]],
) -> Iterable[Optional[float]]:
    """
    Runs a sequence of functions with a given input and returns the results.

    Args:
        functions (Sequence[Callable[[int, bool], float]]): A sequence of functions to run.
        n (int): The input to pass to the functions.
        skip (list[Callable[[int, bool], float]]): A list of functions to skip.

    Yields:
        Optional[float]: The result of each function, or None if the function is in the skip list.
    """

    for f in functions:
        if f in skip:
            yield None
        else:
            yield f(n, False)


def runtests(modules: list[str], optionrange: Sequence[Any], timeout: int) -> None:
    """
    Runs tests for a list of modules with a range of options.

    Args:
        modules (list[str]): A list of module names to test.
        optionrange (Sequence[Any]): A range of options to test.
        timeout (int): The maximum time allowed for a test to run.
                Once a timeout is hit, no further tests are run for
                that specific Module.

    Returns:
        None: This function does not return anything. It prints the test results as comma
              separated values.
    """
    functions = [import_module(module).main for module in modules]
    skip: list[Callable[[int, bool], float]] = []
    print(f"n,{','.join(modules)}")
    for n in optionrange:
        row: list[str] = [str(n)]
        results = run(functions, n, skip)
        for function, time in zip(functions, results):
            if function in skip:
                row.append("")
                continue
            row.append(str(time))
            if timeout is not None and time is not None and time >= timeout:
                skip.append(function)
        print(",".join(row))


if __name__ == "__main__":
    parser = ArgumentParser(
        description="This module runs tests for a list of benchmark modules with a range of "
        "options. The benchmark modules to test must have a `main` function that takes an input "
        "value as an option and a boolean value that specifies whether the benchmark module itself "
        "should produce an output or not.",
        epilog="Example: python -m tests.benchmarks.multi_benchmark -m "
        "tests.benchmarks.benchmark_labyrinth_clsp -m tests.benchmarks.benchmark_labyrinth"
        " -s 5 -e 15 -n 5",
    )
    parser.add_argument(
        "-m",
        "--module",
        action="append",
        default=[],
        required=True,
        help="The module(s) to test. Can be added multiple times.",
    )
    parser.add_argument(
        "-s",
        "--start",
        type=int,
        default=5,
        help="Start value for the range of options. Default is 5.",
    )
    parser.add_argument(
        "-e",
        "--end",
        type=int,
        default=100,
        help="End value for the range of options. Default is 100.",
    )
    parser.add_argument(
        "-n",
        "--step",
        type=int,
        default=5,
        help="Step value for the range of options. Default is 5.",
    )
    parser.add_argument(
        "-t",
        "--timeout",
        type=int,
        default=60,
        help="The maximum time allowed for a test to run. Default is 60",
    )
    args = parser.parse_args()
    functions = args.module
    ns = range(args.start, args.end, args.step)
    timeout = args.timeout
    runtests(functions, ns, timeout)
