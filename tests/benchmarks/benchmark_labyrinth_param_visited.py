from collections.abc import Callable, Mapping
import timeit
from typing import Any
from picls.dsl import DSL
from picls.enumeration import enumerate_terms, interpret_term
from picls.fcl import FiniteCombinatoryLogic

from picls.types import Constructor, Literal, Param, Product, TVar, Type


def plus_one(a: str) -> Callable[[Mapping[str, Literal]], int]:
    def _inner(vars: Mapping[str, Literal]) -> int:
        return int(1 + vars[a].value)

    return _inner


def main(SIZE: int = 10, output: bool = True) -> float:
    def is_free(row: int, col: int) -> bool:
        labyrinth_str = [
            " ┃xxxxxxxx",
            " ┃        ",
            " ┃ ┏━━━━ ┓",
            "   ┃     ┃",
            " ┏━┫ ┏━┓ ┗",
            " ┃ ┃ ┃ ┃  ",
            " ┃ ┃ ┗━┻━ ",
            " ┃ ┃      ",
            " ┗━┛ ┏━━┓ ",
            "     ┃  ┃ ",
        ]
        return labyrinth_str[row][col] == " "
        # SEED = 0
        # if row == col:
        #     return True
        # else:
        #     return (
        #         pow(11, (row + col + SEED) * (row + col + SEED) + col + 7, 1000003) % 5
        #         > 0
        #     )

    U: Callable[
        [int, int, int, tuple[tuple[tuple[int, int], ...], str]],
        tuple[tuple[tuple[int, int], ...], str],
    ] = lambda a, _, c, p: ((*p[0], (c, a)), f"{p[1]} => UP({c}, {a})")
    D: Callable[
        [int, int, int, tuple[tuple[tuple[int, int], ...], str]],
        tuple[tuple[tuple[int, int], ...], str],
    ] = lambda _, b, c, p: ((*p[0], (c, b)), f"{p[1]} => DOWN({c}, {b})")
    L: Callable[
        [int, int, int, tuple[tuple[tuple[int, int], ...], str]],
        tuple[tuple[tuple[int, int], ...], str],
    ] = lambda a, _, c, p: ((*p[0], (a, c)), f"{p[1]} => LEFT({a}, {c})")
    R: Callable[
        [int, int, int, tuple[tuple[tuple[int, int], ...], str]],
        tuple[tuple[tuple[int, int], ...], str],
    ] = lambda _, b, c, p: ((*p[0], (b, c)), f"{p[1]} => RIGHT({b}, {c})")

    pos: Callable[[str, str], Type] = lambda a, b: Constructor(
        "pos", (Product(TVar(a), TVar(b)))
    )

    repo: Mapping[
        Callable[[int, int, int, Any], Any] | Any,
        Param | Type,
    ] = {
        U: DSL()
        .Use("a", "int")
        .Use("b", "int")
        .As(lambda a: a + 1)
        .Use("c", "int")
        .With(lambda c, a: is_free(c, a))
        .Use("pos", pos("c", "b"))
        .With(lambda c, a, pos: (c, a) not in interpret_term(pos)[0])
        .In(pos("c", "a")),
        D: DSL()
        .Use("a", "int")
        .Use("b", "int")
        .As(lambda a: a + 1)
        .Use("c", "int")
        .With(lambda c, b: is_free(c, b))
        .Use("pos", pos("c", "a"))
        .With(lambda c, b, pos: (c, b) not in interpret_term(pos)[0])
        .In(pos("c", "b")),
        L: DSL()
        .Use("a", "int")
        .Use("b", "int")
        .As(lambda a: a + 1)
        .Use("c", "int")
        .With(lambda a, c: is_free(a, c))
        .Use("pos", pos("b", "c"))
        .With(lambda a, c, pos: (a, c) not in interpret_term(pos)[0])
        .In(pos("a", "c")),
        R: DSL()
        .Use("a", "int")
        .Use("b", "int")
        .As(lambda a: a + 1)
        .Use("c", "int")
        .With(lambda b, c: is_free(b, c))
        .Use("pos", pos("a", "c"))
        .With(lambda b, c, pos: (b, c) not in interpret_term(pos)[0])
        .In(pos("b", "c")),
        (((0, 0),), "START"): "pos" @ (Literal(0, "int") * Literal(0, "int")),
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

    fcl: FiniteCombinatoryLogic[
        Callable[[int, int, int, str], str] | str
    ] = FiniteCombinatoryLogic(repo, literals=literals)

    start = timeit.default_timer()
    grammar = fcl.inhabit(fin)

    for i, term in enumerate(enumerate_terms(fin, grammar, 100)):
        t = interpret_term(term)
        if output:
            length = len(t[0])
            print(f"{i}: {length=}, {t[1]}")

    return timeit.default_timer() - start


if __name__ == "__main__":
    main()
