from collections.abc import Callable, Iterable, Sequence
import inspect
from typing import Any, Optional
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


def modulename(function: Any) -> str:
    module = inspect.getmodule(function)
    return module.__name__ if module is not None else "???"


if __name__ == "__main__":
    functions = [
        # benchmark_labyrinth.main,
        benchmark_labyrinth_vanilla_picls.main,
        benchmark_labyrinth_param_pred.main,
    ]
    ns = range(5, 100, 5)
    timeout = 120
    skip: list[Callable[[int, bool], float]] = []
    print(f"n,{','.join(map(modulename, functions))}")
    for n in ns:
        # print(f"{n=}")
        row: list[str] = [str(n)]
        results = run(functions, n, skip)
        for i, (function, time) in enumerate(zip(functions, results)):
            if function in skip:
                # print(f"{modulename}: skipped")
                row.append("-")
                continue
            # print(f"{modulename}: {time}s")
            row.append(str(time))
            if timeout is not None and time is not None and time >= timeout:
                skip.append(function)
        print(",".join(row))
