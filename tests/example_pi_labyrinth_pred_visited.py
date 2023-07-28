from collections.abc import Callable, Mapping
import timeit
from typing import Any
from cls.dsl import Use
from cls.enumeration import enumerate_terms, interpret_term
from cls.fcl import FiniteCombinatoryLogic

from cls.types import Constructor, Literal, Param, Product, TVar, Type


def plus_one(a: str) -> Callable[[Mapping[str, Literal]], int]:
    def _inner(vars: Mapping[str, Literal]) -> int:
        return int(1 + vars[a].value)

    return _inner


def labyrinth(SIZE: int = 10, output: bool = True) -> float:
    def is_free(a: str, b: str) -> Callable[[Mapping[str, Literal]], bool]:
        return lambda vars: _is_free(
            vars[b].value, vars[a].value
        )  # bool(l_str[vars["b"].value][vars["a"].value] == " ")

    def _is_free(row: int, col: int) -> bool:
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
        SEED = 0
        if row == col:
            return True
        else:
            return (
                pow(11, (row + col + SEED) * (row + col + SEED) + col + 7, 1000003) % 5
                > 0
            )

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
        U: Use("a", int)
        .Use("b", int)
        .As(plus_one("a"))
        .Use("c", int)
        .With(is_free("c", "a"))
        .Use("pos", pos("c", "b"))
        .With(
            lambda vars: not isinstance(vars["pos"], Literal)
            and (vars["c"].value, vars["a"].value) not in interpret_term(vars["pos"])[0]
        )
        .In(pos("c", "a")),
        D: Use("a", int)
        .Use("b", int)
        .As(plus_one("a"))
        .Use("c", int)
        .With(is_free("c", "b"))
        .Use("pos", pos("c", "a"))
        .With(
            lambda vars: not isinstance(vars["pos"], Literal)
            and (vars["c"].value, vars["b"].value) not in interpret_term(vars["pos"])[0]
        )
        .In(pos("c", "b")),
        L: Use("a", int)
        .Use("b", int)
        .As(plus_one("a"))
        .Use("c", int)
        .With(is_free("a", "c"))
        .Use("pos", pos("b", "c"))
        .With(
            lambda vars: not isinstance(vars["pos"], Literal)
            and (vars["a"].value, vars["c"].value) not in interpret_term(vars["pos"])[0]
        )
        .In(pos("a", "c")),
        R: Use("a", int)
        .Use("b", int)
        .As(plus_one("a"))
        .Use("c", int)
        .With(is_free("b", "c"))
        .Use("pos", pos("a", "c"))
        .With(
            lambda vars: not isinstance(vars["pos"], Literal)
            and (vars["b"].value, vars["c"].value) not in interpret_term(vars["pos"])[0]
        )
        .In(pos("b", "c")),
        (((0, 0),), "START"): "pos" @ (Literal(0, int) * Literal(0, int)),
    }

    literals = {int: list(range(SIZE))}

    if output:
        for row in range(SIZE):
            for col in range(SIZE):
                if _is_free(row, col):
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

    for i, term in enumerate(enumerate_terms(fin, grammar, 100)):
        t = interpret_term(term)
        if output:
            length = len(t[0])
            print(f"{i}: {length=}, {t[1]}")

    return timeit.default_timer() - start


if __name__ == "__main__":
    labyrinth()
