from functools import partial, cache

from collections.abc import Callable, Mapping
import timeit
from itertools import product
from typing import Any, cast
from clsp.dsl import DSL
from clsp.enumeration import Tree, enumerate_terms, interpret_term
from clsp.fcl import FiniteCombinatoryLogic

from clsp.types import Constructor, Literal, Param, LVar, Type


@cache
def visited(path: Tree[Any]) -> set[tuple[int, int]]:
    if path.root == "START":
        return {(0, 0)}
    return {cast(tuple[int, int], path.parameters["a"].root)} | visited(path.parameters["pos"])


def is_free(size: int, pos: tuple[int, int]) -> bool:
    """
    Create a maze in the form:
    XXX...XXX
    X       X
    X X...X X
    . ..... .
    X X...X X
    X       X
    X X...XXX
    X       X
    XXX...XXX
    """

    col, row = pos
    if row in [0, size - 1, size - 3]:
        return True
    else:
        if row == size - 2 and col == size - 1:
            return False
        if col in [0, size - 1]:
            return True
        return False


def main(SIZE: int = 10, output: bool = True) -> float:
    visited.cache_clear()
    U: Callable[[int, int, str], str] = lambda b, _, p: f"{p} => UP({b})"
    D: Callable[[int, int, str], str] = lambda b, _, p: f"{p} => DOWN({b})"
    L: Callable[[int, int, str], str] = lambda b, _, p: f"{p} => LEFT({b})"
    R: Callable[[int, int, str], str] = lambda b, _, p: f"{p} => RIGHT({b})"

    pos: Callable[[str], Type] = lambda ab: Constructor("pos", (LVar(ab)))

    repo: Mapping[
        Callable[[int, int, str], str] | str,
        Param | Type,
    ] = {
        U: DSL()
        .Use("b", "int2")
        .Use("a", "int2")
        .As(lambda b: (b[0], b[1] + 1))
        .Use("pos", pos("a"))
        .With(lambda b, pos: b not in visited(pos))
        .In(pos("b")),
        D: DSL()
        .Use("b", "int2")
        .Use("a", "int2")
        .As(lambda b: (b[0], b[1] - 1))
        .Use("pos", pos("a"))
        .With(lambda b, pos: b not in visited(pos))
        .In(pos("b")),
        L: DSL()
        .Use("b", "int2")
        .Use("a", "int2")
        .As(lambda b: (b[0] + 1, b[1]))
        .Use("pos", pos("a"))
        .With(lambda b, pos: b not in visited(pos))
        .In(pos("b")),
        R: DSL()
        .Use("b", "int2")
        .Use("a", "int2")
        .As(lambda b: (b[0] - 1, b[1]))
        .Use("pos", pos("a"))
        .With(lambda b, pos: b not in visited(pos))
        .In(pos("b")),
        "START": "pos" @ (Literal((0, 0), "int2")),
    }

    # literals = {"int": list(range(SIZE))}
    literals = {"int2": list(filter(partial(is_free, SIZE), product(range(SIZE), range(SIZE))))}

    if output:
        for row in range(SIZE):
            for col in range(SIZE):
                if is_free(SIZE, (col, row)):
                    print("-", end="")
                else:
                    print("#", end="")
            print("")

    fin = "pos" @ (Literal((SIZE - 1, SIZE - 1), "int2"))

    fcl: FiniteCombinatoryLogic[Callable[[int, int, str], str] | str] = FiniteCombinatoryLogic(
        repo, literals=literals
    )

    start = timeit.default_timer()
    grammar = fcl.inhabit(fin)

    for term in enumerate_terms(fin, grammar, 3):
        t = interpret_term(term)
        if output:
            print(t)

    return timeit.default_timer() - start


if __name__ == "__main__":
    main()
