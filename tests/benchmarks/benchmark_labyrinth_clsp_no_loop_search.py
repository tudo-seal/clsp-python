from collections.abc import Callable, Iterable, Mapping
import timeit
from itertools import product
from typing import Any

from clsp.dsl import DSL
from clsp.enumeration import Tree, enumerate_terms, interpret_term
from clsp.fcl import FiniteCombinatoryLogic

from clsp.types import Constructor, Literal, Param, LVar, Type

from clsp.search import TournamentSelection, SimpleEA


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


def getpath(
    path: Tree[Any, Any]
) -> Iterable[tuple[int, int]]:
    position_arg = path.parameters["b"].root
    while path.root != "START":
        if isinstance(position_arg, tuple):
            yield position_arg
        position_arg = path.parameters["a"].root
        path = path.parameters["pos"]


def main(solutions: int = 10000, output: bool = True) -> float:
    SIZE = 5
    U: Callable[[int, int, str], str] = lambda _, b, p: f"{p} => UP({b})"
    D: Callable[[int, int, str], str] = lambda _, b, p: f"{p} => DOWN({b})"
    L: Callable[[int, int, str], str] = lambda _, b, p: f"{p} => LEFT({b})"
    R: Callable[[int, int, str], str] = lambda _, b, p: f"{p} => RIGHT({b})"

    pos: Callable[[str], Type] = lambda ab: Constructor("pos", (LVar(ab)))

    repo: Mapping[
        Callable[[int, int, str], str] | str,
        Param | Type,
    ] = {
        U: DSL()
        .Use("a", "int2")
        .Use("b", "int2")
        .With(lambda b: is_free(b))
        .As(lambda a: (a[0], a[1] - 1))
        .Use("pos", pos("a"))
        .In(pos("b")),
        D: DSL()
        .Use("a", "int2")
        .Use("b", "int2")
        .With(lambda b: is_free(b))
        .As(lambda a: (a[0], a[1] + 1))
        .Use("pos", pos("a"))
        .In(pos("b")),
        L: DSL()
        .Use("a", "int2")
        .Use("b", "int2")
        .With(lambda b: is_free(b))
        .As(lambda a: (a[0] - 1, a[1]))
        .Use("pos", pos("a"))
        .In(pos("b")),
        R: DSL()
        .Use("a", "int2")
        .Use("b", "int2")
        .With(lambda b: is_free(b))
        .As(lambda a: (a[0] + 1, a[1]))
        .Use("pos", pos("a"))
        .In(pos("b")),
        "START": "pos" @ (Literal((0, 0), "int2")),
    }

    # literals = {"int": list(range(SIZE))}
    literals = {"int2": list(product(range(SIZE), range(SIZE)))}

    if output:
        for row in range(SIZE):
            for col in range(SIZE):
                if is_free((row, col)):
                    print(f"-", end="")
                else:
                    print("#", end="")
            print("")

    fin = "pos" @ (Literal((SIZE - 1, SIZE - 1), "int2"))

    fcl: FiniteCombinatoryLogic[Callable[[int, int, str], str] | str] = FiniteCombinatoryLogic(
        repo, literals=literals
    )

    start = timeit.default_timer()
    grammar = fcl.inhabit(fin)

    def shortest_loop_free_path(tree: Tree[Any, Any]) -> int:
        path = list(getpath(tree))
        length = len(path)
        if length != len(set(path)):
            return -100000000
        else:
            return length*(-1)

    def longest_loop_free_path(tree: Tree[Any, Any]) -> int:
        path = list(getpath(tree))
        length = len(path)
        if length != len(set(path)):
            return -1
        else:
            return length

    tournament_selection = TournamentSelection(3, 1000)

    tournament_search = SimpleEA(repo, literals, fin, selection_strategy=tournament_selection, generations=4).search_fittest


    final_population = list(tournament_search(longest_loop_free_path, 500)) # 500 overwrites 1000 from above

    #for term in final_population[:10]:
    #    positions = list(getpath(term))
    #    t = interpret_term(term)
    #    print(t)
    #    print(f"Path length: {len(positions)}")
    #    print("#######################################")

    print("maximum:")
    winner = final_population[0]
    print(interpret_term(winner))
    win_path = list(getpath(winner))
    print(f"Path: {win_path}")
    print(f"Path set length: {len(set(win_path))}")
    return timeit.default_timer() - start


if __name__ == "__main__":
    main()
