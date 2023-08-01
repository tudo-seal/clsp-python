from collections.abc import Callable, Iterable, Sequence
import inspect
from typing import Optional
from . import benchmark_labyrinth
from . import benchmark_labyrinth_vanilla_picls
from . import benchmark_labyrinth_param_pred


def run(
    functions: Sequence[Callable[[int, bool], float]],
    n: int,
    skip: list[Callable[[int, bool], float]],
) -> Iterable[Optional[float]]:
    for f in functions:
        if f in skip:
            yield None
        else:
            yield f(n, False)


if __name__ == "__main__":
    functions = [
        benchmark_labyrinth.main,
        benchmark_labyrinth_vanilla_picls.main,
        benchmark_labyrinth_param_pred.main,
    ]
    ns = range(5, 60, 5)
    timeout = 20
    skip: list[Callable[[int, bool], float]] = []
    for n in ns:
        print(f"{n=}")
        results = run(functions, n, skip)
        for i, (function, time) in enumerate(zip(functions, results)):
            module = inspect.getmodule(function)
            modulename = module.__name__ if module is not None else "???"
            if function in skip:
                print(f"{modulename}: skipped")
                continue
            print(f"{modulename}: {time}s")
            if timeout is not None and time is not None and time >= timeout:
                skip.append(function)
