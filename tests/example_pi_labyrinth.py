from collections.abc import Callable, Mapping
from typing import Any
from cls.dsl import Requires, Use
from cls.enumeration import enumerate_terms, interpret_term
from cls.fcl import FiniteCombinatoryLogic

from cls.types import Literal, Param, TVar, Type


def pred_plus_one(a: str, b: str) -> Callable[[Mapping[str, Literal]], bool]:
    def _inner(vars: Mapping[str, Literal]) -> bool:
        return bool(vars[a].value == vars[b].value + 1)

    return _inner


def labyrinth() -> None:
    def make_is_free(l_str: list[str]) -> Callable[[Mapping[str, Literal]], bool]:
        return lambda vars: is_free(
            vars["b"].value, vars["a"].value
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

    FREE = lambda a, b: f"FREE({a}, {b})"
    U = lambda a, b, c, p, f: f"{p} => UP({c}, {a})"
    D = lambda a, b, c, p, f: f"{p} => DOWN({c}, {b})"
    L = lambda a, b, c, p, f: f"{p} => LEFT({a}, {c})"
    R = lambda a, b, c, p, f: f"{p} => RIGHT({b}, {c})"

    free: Callable[[str, str], Type[str]] = lambda a, b: "free" @ (TVar(a) * TVar(b))
    pos: Callable[[str, str], Type[str]] = lambda a, b: "pos" @ (TVar(a) * TVar(b))

    repo: dict[
        Callable[[Any, Any, Any, Any, Any], str] | Callable[[Any, Any], str] | str,
        Param[str] | Type[str],
    ] = {
        FREE: Use("a", int)
        .Use("b", int)
        .With(make_is_free(labyrinth_str))
        .In(free("a", "b")),
        U: Use("a", int)
        .Use("b", int)
        .With(pred_plus_one("b", "a"))
        .Use("c", int)
        .In(Requires(pos("c", "b"), free("c", "a")).Provides(pos("c", "a"))),
        D: Use("a", int)
        .Use("b", int)
        .With(pred_plus_one("b", "a"))
        .Use("c", int)
        .In(pos("c", "a") ** free("c", "b") ** pos("c", "b")),
        L: Use("a", int)
        .Use("b", int)
        .With(pred_plus_one("b", "a"))
        .Use("c", int)
        .In(pos("b", "c") ** free("a", "c") ** pos("a", "c")),
        R: Use("a", int)
        .Use("b", int)
        .With(pred_plus_one("b", "a"))
        .Use("c", int)
        .In(pos("a", "c") ** free("b", "c") ** pos("b", "c")),
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
