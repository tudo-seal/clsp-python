from collections.abc import Callable, Iterable, Mapping
import timeit
from itertools import product
from clsp.dsl import DSL
from clsp.tree import Tree
from clsp.synthesizer import Synthesizer, Specification
from typing import Any

from clsp.types import Constructor, Literal, Var, Type

def is_free(pos: tuple[int, int]) -> bool:
    col, row = pos
    SEED = 0
    if row == col:
        return True
    else:
        return pow(11, (row + col + SEED) * (row + col + SEED) + col + 7, 1000003) % 5 > 0


def getpath(
    path: Tree[Any]
) -> Iterable[tuple[int, int]]:
    position_arg = path.parameters["b"].root
    while path.root != "START":
        if isinstance(position_arg, tuple):
            yield position_arg
        position_arg = path.parameters["a"].root
        path = path.parameters["pos"]


def main(solutions: int = 10000, output: bool = True) -> float:
    SIZE = 4
    U: Callable[[int, int, str], str] = lambda _, b, p: f"{p} => UP({b})"
    D: Callable[[int, int, str], str] = lambda _, b, p: f"{p} => DOWN({b})"
    L: Callable[[int, int, str], str] = lambda _, b, p: f"{p} => LEFT({b})"
    R: Callable[[int, int, str], str] = lambda _, b, p: f"{p} => RIGHT({b})"

    pos: Callable[[str], Type] = lambda ab: Constructor("pos", (Var(ab)))

    componentSpeficifations: Mapping[
        Callable[[int, int, str], str] | str,
        Specification,
    ] = {
        U: DSL()
        .Use("a", "int2")
        .Use("b", "int2", lambda vars: [(vars["a"][0], vars["a"][1] - 1)])
        .SuchThat(lambda vars: is_free(vars["b"]))
        .Use("pos", pos("a"))
        .In(pos("b")),
        D: DSL()
        .Use("a", "int2")
        .Use("b", "int2", lambda vars: [(vars["a"][0], vars["a"][1] + 1)])
        .SuchThat(lambda vars: is_free(vars["b"]))
        .Use("pos", pos("a"))
        .In(pos("b")),
        L: DSL()
        .Use("a", "int2")
        .Use("b", "int2", lambda vars: [(vars["a"][0] - 1, vars["a"][1])])
        .SuchThat(lambda vars: is_free(vars["b"]))
        .Use("pos", pos("a"))
        .In(pos("b")),
        R: DSL()
        .Use("a", "int2")
        .Use("b", "int2", lambda vars: [(vars["a"][0] + 1, vars["a"][1])])
        .SuchThat(lambda vars: is_free(vars["b"]))
        .Use("pos", pos("a"))
        .In(pos("b")),
        "START": "pos" @ (Literal((0, 0), "int2")),
    }

    literals = {"int2": list(product(range(SIZE), range(SIZE)))}

    if output:
        for row in range(SIZE):
            for col in range(SIZE):
                if is_free((col, row)):
                    print("-", end="")
                else:
                    print("#", end="")
            print("")

    fin = "pos" @ (Literal((SIZE - 1, SIZE - 1), "int2"))

    synthesizer: Synthesizer[Callable[[int, int, str], str] | str] = Synthesizer(
        componentSpeficifations, parameterSpace=literals
    )

    start = timeit.default_timer()
    grammar = synthesizer.constructSolutionSpace(fin)

    for tree in grammar.enumerate_trees(fin, solutions):
        positions = list(getpath(tree))
        if len(positions) != len(set(positions)):
            continue

        if output:
            print(tree.interpret())

    return timeit.default_timer() - start


if __name__ == "__main__":
    main()
