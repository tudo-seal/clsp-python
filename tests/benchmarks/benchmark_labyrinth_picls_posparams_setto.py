from collections.abc import Callable, Mapping
import timeit
from typing import Any
from picls.dsl import Use
from picls.enumeration import enumerate_terms, interpret_term
from picls.fcl import FiniteCombinatoryLogic

from picls.types import Constructor, Literal, Param, Product, TVar, Type


def plus_one(a: str) -> Callable[[Mapping[str, Literal]], int]:
    def _inner(vars: Mapping[str, Literal]) -> int:
        return int(1 + vars[a].value)

    return _inner


def main(SIZE: int = 10, output: bool = True) -> float:
    def is_free(col: int, row: int) -> bool:
        SEED = 0
        if row == col:
            return True
        else:
            return (
                pow(11, (row + col + SEED) * (row + col + SEED) + col + 7, 1000003) % 5
                > 0
            )

    FREE = lambda a, b: f"FREE({a}, {b})"
    U = lambda a, _, c, p, f: f"{p} => UP({c}, {a})"
    D = lambda _, b, c, p, f: f"{p} => DOWN({c}, {b})"
    L = lambda a, _, c, p, f: f"{p} => LEFT({a}, {c})"
    R = lambda _, b, c, p, f: f"{p} => RIGHT({b}, {c})"

    pos: Callable[[str, str], Type] = lambda a, b: Constructor(
        "pos", (Product(TVar(a), TVar(b)))
    )
    free: Callable[[str, str], Type] = lambda a, b: Constructor(
        "free", Product(TVar(a), TVar(b))
    )

    repo: Mapping[
        Callable[[int, int, int, str], str] | str | Any,
        Param | Type,
    ] = {
        FREE: Use("a", int)
        .Use("b", int)
        .With(lambda a, b: is_free(b, a))
        .In(free("a", "b")),
        U: Use("a", int)
        .Use("b", int)
        .As(lambda a: a + 1)
        .Use("c", int)
        .Use("pos", pos("c", "b"))
        .In(free("c", "a") ** pos("c", "a")),
        D: Use("a", int)
        .Use("b", int)
        .As(lambda a: a + 1)
        .Use("c", int)
        .Use("pos", pos("c", "a"))
        .In(free("c", "b") ** pos("c", "b")),
        L: Use("a", int)
        .Use("b", int)
        .As(lambda a: a + 1)
        .Use("c", int)
        .Use("pos", pos("b", "c"))
        .In(free("a", "c") ** pos("a", "c")),
        R: Use("a", int)
        .Use("b", int)
        .As(lambda a: a + 1)
        .Use("c", int)
        .Use("pos", pos("a", "c"))
        .In(free("b", "c") ** pos("b", "c")),
        "START": "pos" @ (Literal(0, int) * Literal(0, int)),
    }

    literals = {int: list(range(SIZE))}

    if output:
        for row in range(SIZE):
            for col in range(SIZE):
                if is_free(row, col):
                    print("-", end="")
                else:
                    print("#", end="")
            print("")

    fin = "pos" @ (Literal(SIZE - 1, int) * Literal(SIZE - 1, int))

    fcl: FiniteCombinatoryLogic[
        Callable[[int, int, int, str], str] | str
    ] = FiniteCombinatoryLogic(repo, literals=literals)

    start = timeit.default_timer()
    grammar = fcl.inhabit(fin)

    for term in enumerate_terms(fin, grammar, 3):
        t = interpret_term(term)
        if output:
            print(t)

    return timeit.default_timer() - start


if __name__ == "__main__":
    main()
