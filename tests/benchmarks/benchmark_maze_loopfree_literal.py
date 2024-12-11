import timeit
from collections.abc import Callable, Iterable, Mapping
from functools import partial
from itertools import chain, combinations, product
from typing import Any, cast

from clsp.dsl import DSL
from clsp.enumeration import Tree, enumerate_terms, interpret_term
from clsp.fcl import FiniteCombinatoryLogic
from clsp.types import Constructor, Literal, LVar, Param, Type


def startc(visited: set[tuple[int, int]]) -> str:
    return "START"


def visited(path: Tree[Any]) -> set[tuple[int, int]]:
    if path.root == startc:
        return {(0, 0)}
    return {cast(tuple[int, int], path.parameters["a"].root)} | visited(path.parameters["pos"])


def powerset(s: list[tuple[int, int]]) -> list[frozenset[tuple[int, int]]]:
    return list(map(frozenset, chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))))


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
    U: Callable[..., str] = lambda _old_visited, b, _a, _new_visited, p: f"{p} => UP({b})"
    D: Callable[..., str] = lambda _old_visited, b, _a, _new_visited, p: f"{p} => DOWN({b})"
    L: Callable[..., str] = lambda _old_visited, b, _a, _new_visited, p: f"{p} => LEFT({b})"
    R: Callable[..., str] = lambda _old_visited, b, _a, _new_visited, p: f"{p} => RIGHT({b})"

    pos: Callable[[str], Type] = lambda ab: Constructor("pos", (LVar(ab)))
    vis: Callable[[str], Type] = lambda ab: Constructor("vis", (LVar(ab)))

    repo: Mapping[
        Callable[..., str] | str,
        Param | Type,
    ] = {
        U: DSL()
        .Use("old_visited", "power_int2")
        .Use("b", "int2")
        .With(lambda b, old_visited: b not in old_visited)
        .Use("a", "int2")
        .As(lambda b: (b[0], b[1] + 1))
        .Use("new_visited", "power_int2")
        .As(lambda b, old_visited: old_visited | {b})
        .Use("pos", pos("a") & vis("new_visited"))
        .In(pos("b") & vis("old_visited")),
        D: DSL()
        .Use("old_visited", "power_int2")
        .Use("b", "int2")
        .With(lambda b, old_visited: b not in old_visited)
        .Use("a", "int2")
        .As(lambda b: (b[0], b[1] - 1))
        .Use("new_visited", "power_int2")
        .As(lambda b, old_visited: old_visited | {b})
        .Use("pos", pos("a") & vis("new_visited"))
        .In(pos("b") & vis("old_visited")),
        L: DSL()
        .Use("old_visited", "power_int2")
        .Use("b", "int2")
        .With(lambda b, old_visited: b not in old_visited)
        .Use("a", "int2")
        .As(lambda b: (b[0] + 1, b[1]))
        .Use("new_visited", "power_int2")
        .As(lambda b, old_visited: old_visited | {b})
        .Use("pos", pos("a") & vis("new_visited"))
        .In(pos("b") & vis("old_visited")),
        R: DSL()
        .Use("old_visited", "power_int2")
        .Use("b", "int2")
        .With(lambda b, old_visited: b not in old_visited)
        .Use("a", "int2")
        .As(lambda b: (b[0] - 1, b[1]))
        .Use("new_visited", "power_int2")
        .As(lambda b, old_visited: old_visited | {b})
        .Use("pos", pos("a") & vis("new_visited"))
        .In(pos("b") & vis("old_visited")),
        startc: DSL()
        .Use("visited", "power_int2")
        .With(lambda visited: (0, 0) not in visited)
        .In("pos" @ (Literal((0, 0), "int2")) & vis("visited")),
    }

    # literals = {"int": list(range(SIZE))}

    if output:
        for row in range(SIZE):
            for col in range(SIZE):
                if is_free(SIZE, (col, row)):
                    print("-", end="")
                else:
                    print("#", end="")
            print("")
    positions = list(filter(partial(is_free, SIZE), product(range(SIZE), range(SIZE))))
    power_positions = powerset(positions)
    literals = {"int2": positions, "power_int2": power_positions}

    fin = ("pos" @ (Literal((SIZE - 1, SIZE - 1), "int2"))) & (
        "vis" @ Literal(frozenset(), "power_int2")
    )

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
