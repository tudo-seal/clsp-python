from collections.abc import Callable, Mapping
from typing import Any
from cls.dsl import Requires, Use
from cls.enumeration import enumerate_terms, interpret_term
from cls.fcl import FiniteCombinatoryLogic

from cls.types import Literal, Param, TVar, Type


def plus_one(a: str) -> Callable[[Mapping[str, Literal]], int]:
    def _inner(vars: Mapping[str, Literal]) -> bool:
        return 1 + vars[a].value

    return _inner


def labyrinth() -> None:
    def make_is_free(a, b) -> Callable[[Mapping[str, Literal]], bool]:
        return lambda vars: is_free(
            vars[b].value, vars[a].value
        )  # bool(l_str[vars["b"].value][vars["a"].value] == " ")

    def is_free(row: int, col: int) -> bool:
        SEED = 0
        if row == col:
            return True
        else:
            return (
                pow(11, (row + col + SEED) * (row + col + SEED) + col + 7, 1000003) % 5
                > 0
            )

    labyrinth_str = [
        " ┃        ",
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

    U = lambda a, b, c, p: f"{p} => UP({c}, {a})"
    D = lambda a, b, c, p: f"{p} => DOWN({c}, {b})"
    L = lambda a, b, c, p: f"{p} => LEFT({a}, {c})"
    R = lambda a, b, c, p: f"{p} => RIGHT({b}, {c})"

    pos: Callable[[str, str], Type[str]] = lambda a, b: "pos" @ (TVar(a) * TVar(b))

    repo: dict[
        Callable[[Any, Any, Any, Any], str] | Callable[[Any, Any], str] | str,
        Param[str] | Type[str],
    ] = {
        U: Use("a", int)
        .Use("b", int)
        .As(plus_one("a"))
        .Use("c", int)
        .With(make_is_free("c", "a"))
        .Use("pos", pos("c", "b"))
        .In(pos("c", "a")),
        D: Use("a", int)
        .Use("b", int)
        .As(plus_one("a"))
        .Use("c", int)
        .With(make_is_free("c", "b"))
        .Use("pos", pos("c", "a"))
        .In(pos("c", "b")),
        L: Use("a", int)
        .Use("b", int)
        .As(plus_one("a"))
        .Use("c", int)
        .With(make_is_free("a", "c"))
        .Use("pos", pos("a", "b"))
        .In(pos("a", "c")),
        R: Use("a", int)
        .Use("b", int)
        .As(plus_one("a"))
        .Use("c", int)
        .With(make_is_free("b", "c"))
        .Use("pos", pos("a", "c"))
        .In(pos("b", "c")),
        "START": "pos" @ (Literal(0, int) * Literal(0, int)),
    }

    SIZE = 10

    literals = {int: list(range(SIZE))}

    # print("▒▒▒▒▒▒▒▒▒▒▒▒")
    # for line in labyrinth_str:
    #     print(f"▒{line}▒")
    # print("▒▒▒▒▒▒▒▒▒▒▒▒")
    for row in range(SIZE):
        for col in range(SIZE):
            if is_free(row, col):
                print("-", end="")
            else:
                print("#", end="")
        print("")

    fin = "pos" @ (Literal(SIZE - 1, int) * Literal(SIZE - 1, int))

    fcl: FiniteCombinatoryLogic[
        str, Callable[[Any, Any, Any, Any, Any], str] | Callable[[Any, Any], str] | str
    ] = FiniteCombinatoryLogic(repo, literals=literals)

    grammar = fcl.inhabit(fin)

    for term in enumerate_terms(fin, grammar, 3):
        print(interpret_term(term))


if __name__ == "__main__":
    labyrinth()
