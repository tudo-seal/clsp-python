from collections.abc import Callable, Mapping
import timeit
from itertools import product
from clsp.dsl import DSL
from clsp.synthesizer import Synthesizer, Specification

from clsp.types import Constructor, Literal, Type, Var


def plus_one(a: str) -> Callable[[Mapping[str, Literal]], int]:
    def _inner(vars: Mapping[str, Literal]) -> int:
        return int(1 + vars[a].value)

    return _inner


def is_free(pos: tuple[int, int]) -> bool:
    col, row = pos
    SEED = 0
    if row == col:
        return True
    else:
        return pow(11, (row + col + SEED) * (row + col + SEED) + col + 7, 1000003) % 5 > 0


def main(SIZE: int = 30, output: bool = False) -> float:
    U: Callable[[int, int, str], str] = lambda b, _, p: f"{p} => UP({b})"
    D: Callable[[int, int, str], str] = lambda b, _, p: f"{p} => DOWN({b})"
    L: Callable[[int, int, str], str] = lambda b, _, p: f"{p} => LEFT({b})"
    R: Callable[[int, int, str], str] = lambda b, _, p: f"{p} => RIGHT({b})"

    pos: Callable[[str], Type] = lambda ab: Constructor("pos", (Var(ab)))

    repo: Mapping[
        Callable[[int, int, str], str] | str,
        Specification,
    ] = {
        U: DSL()
        .Parameter("a", "int2")
        .Parameter("b", "int2", lambda vars: [(vars["a"][0], vars["a"][1] - 1)])
        .Argument("pos", pos("a"))
        .Suffix(pos("b")),
        D: DSL()
        .Parameter("a", "int2")
        .Parameter("b", "int2", lambda vars: [(vars["a"][0], vars["a"][1] + 1)])
        .Argument("pos", pos("a"))
        .Suffix(pos("b")),
        L: DSL()
        .Parameter("a", "int2")
        .Parameter("b", "int2", lambda vars: [(vars["a"][0] - 1, vars["a"][1])])
        .Argument("pos", pos("a"))
        .Suffix(pos("b")),
        R: DSL()
        .Parameter("a", "int2")
        .Parameter("b", "int2", lambda vars: [(vars["a"][0] + 1, vars["a"][1])])
        .Argument("pos", pos("a"))
        .Suffix(pos("b")),
        "START": "pos" @ (Literal((0, 0), "int2")),
    }

    # literals = {"int": list(range(SIZE))}
    literals = {"int2": list(filter(is_free, product(range(SIZE), range(SIZE))))}

    if output:
        for row in range(SIZE):
            for col in range(SIZE):
                if is_free((row, col)):
                    print("-", end="")
                else:
                    print("#", end="")
            print("")

    fin = "pos" @ (Literal((SIZE - 1, SIZE - 1), "int2"))

    synthesizer: Synthesizer[Callable[[int, int, str], str] | str] = Synthesizer(
        repo, literals
    )

    start = timeit.default_timer()
    grammar = synthesizer.constructSolutionSpace(fin)
    print("finished constructing solution space")
    for term in grammar.enumerate_trees(fin, 3):
        t = term.interpret()
        if output:
            print(t)

    print(timeit.default_timer() - start)
    return timeit.default_timer() - start


if __name__ == "__main__":
    main()
