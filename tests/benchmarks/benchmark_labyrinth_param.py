from collections.abc import Callable
import timeit
from typing import Any
from picls.dsl import Requires, Use
from picls.enumeration import enumerate_terms, interpret_term
from picls.fcl import FiniteCombinatoryLogic

from picls.types import Literal, Param, TVar, Type


def main(SIZE: int = 10, output: bool = True) -> float:
    def is_free(row: int, col: int) -> bool:
        SEED = 0
        if row == col:
            return True
        else:
            return (
                pow(11, (row + col + SEED) * (row + col + SEED) + col + 7, 1000003) % 5
                > 0
            )

    FREE = lambda a, b: f"FREE({a}, {b})"
    U = lambda a, b, c, p, f: f"{p} => UP({c}, {a})"
    D = lambda a, b, c, p, f: f"{p} => DOWN({c}, {b})"
    L = lambda a, b, c, p, f: f"{p} => LEFT({a}, {c})"
    R = lambda a, b, c, p, f: f"{p} => RIGHT({b}, {c})"

    free: Callable[[str, str], Type] = lambda a, b: "free" @ (TVar(a) * TVar(b))
    pos: Callable[[str, str], Type] = lambda a, b: "pos" @ (TVar(a) * TVar(b))

    repo: dict[
        Callable[[Any, Any, Any, Any, Any], str] | Callable[[Any, Any], str] | str,
        Param | Type,
    ] = {
        FREE: Use("a", int)
        .Use("b", int)
        .With(lambda a, b: is_free(b, a))
        .In(free("a", "b")),
        U: Use("a", int)
        .Use("b", int)
        .With(lambda a, b: a == b + 1)
        .Use("c", int)
        .In(Requires(pos("c", "b"), free("c", "a")).Provides(pos("c", "a"))),
        D: Use("a", int)
        .Use("b", int)
        .With(lambda a, b: a == b + 1)
        .Use("c", int)
        .In(pos("c", "a") ** free("c", "b") ** pos("c", "b")),
        L: Use("a", int)
        .Use("b", int)
        .With(lambda a, b: a == b + 1)
        .Use("c", int)
        .In(pos("b", "c") ** free("a", "c") ** pos("a", "c")),
        R: Use("a", int)
        .Use("b", int)
        .With(lambda a, b: a == b + 1)
        .Use("c", int)
        .In(pos("a", "c") ** free("b", "c") ** pos("b", "c")),
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
        Callable[[Any, Any, Any, Any, Any], str] | Callable[[Any, Any], str] | str
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
