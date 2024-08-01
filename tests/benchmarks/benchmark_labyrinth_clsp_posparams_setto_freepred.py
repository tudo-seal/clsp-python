from collections.abc import Callable, Mapping
import timeit
from clsp.dsl import DSL
from clsp.enumeration import enumerate_terms, interpret_term
from clsp.fcl import FiniteCombinatoryLogic

from clsp.types import Constructor, Literal, Param, Product, LVar, Type


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

    U: Callable[[int, int, int, str], str] = lambda a, _, c, p: f"{p} => UP({c}, {a})"
    D: Callable[[int, int, int, str], str] = lambda _, b, c, p: f"{p} => DOWN({c}, {b})"
    L: Callable[[int, int, int, str], str] = lambda a, _, c, p: f"{p} => LEFT({a}, {c})"
    R: Callable[[int, int, int, str], str] = (
        lambda _, b, c, p: f"{p} => RIGHT({b}, {c})"
    )

    pos: Callable[[str, str], Type] = lambda a, b: Constructor(
        "pos", (Product(LVar(a), LVar(b)))
    )

    repo: Mapping[
        Callable[[int, int, int, str], str] | str,
        Param | Type,
    ] = {
        U: DSL()
        .Use("a", "int")
        .Use("b", "int")
        .As(lambda a: a + 1)
        .Use("c", "int")
        .With(lambda c, a: is_free(c, a))
        .Use("pos", pos("c", "b"))
        .In(pos("c", "a")),
        D: DSL()
        .Use("a", "int")
        .Use("b", "int")
        .As(lambda a: a + 1)
        .Use("c", "int")
        .With(lambda c, b: is_free(c, b))
        .Use("pos", pos("c", "a"))
        .In(pos("c", "b")),
        L: DSL()
        .Use("a", "int")
        .Use("b", "int")
        .As(lambda a: a + 1)
        .Use("c", "int")
        .With(lambda a, c: is_free(a, c))
        .Use("pos", pos("b", "c"))
        .In(pos("a", "c")),
        R: DSL()
        .Use("a", "int")
        .Use("b", "int")
        .As(lambda a: a + 1)
        .Use("c", "int")
        .With(lambda b, c: is_free(b, c))
        .Use("pos", pos("a", "c"))
        .In(pos("b", "c")),
        "START": "pos" @ (Literal(0, "int") * Literal(0, "int")),
    }

    literals = {"int": list(range(SIZE))}

    if output:
        for row in range(SIZE):
            for col in range(SIZE):
                if is_free(row, col):
                    print("-", end="")
                else:
                    print("#", end="")
            print("")

    fin = "pos" @ (Literal(SIZE - 1, "int") * Literal(SIZE - 1, "int"))

    fcl: FiniteCombinatoryLogic[Callable[[int, int, int, str], str] | str] = (
        FiniteCombinatoryLogic(repo, literals=literals)
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
