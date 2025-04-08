from collections.abc import Callable, Iterable, Mapping
import timeit
from itertools import product
from clsp.dsl import DSL
from clsp.enumeration import Tree, enumerate_terms, interpret_term
from clsp.fcl import FiniteCombinatoryLogic

from clsp.types import Constructor, Literal, Param, LVar, Type

from clsp.search import tournament_search, Fitness


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
    path
) -> Iterable[tuple[int, int]]:
    while path.root != "START":
        # TODO why do empty parameters occur? This shouldn't happen, should it?
        if not path.parameters:
            break
        position_arg = path.parameters["a"].root
        if isinstance(position_arg, tuple):
            yield position_arg
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

    def shortest_loop_free_path(tree) -> int:
        path = list(getpath(tree))
        length = len(path)
        if len(path) != len(set(path)):
            return -100000000
        else:
            return length*(-1)

    def longest_loop_free_path(tree) -> int:
        path = list(getpath(tree))
        length = len(path)
        if len(path) != len(set(path)):
            return -1
        else:
            return length

    fit = Fitness(shortest_loop_free_path, "shortest_path_and_loop_free", ordering=lambda x, y: x < y)

    for term in list(tournament_search(fin, grammar, fit, population_size=100, generations=1, tournament_size=3, preserved_fittest=1))[:10]:
        positions = list(getpath(term))
        t = interpret_term(term)
        print(t)
        print(f"Path length: {len(positions)}")
        print("#######################################")

    return timeit.default_timer() - start


if __name__ == "__main__":
    main()
