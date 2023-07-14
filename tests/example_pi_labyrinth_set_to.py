from collections.abc import Callable
from typing import Any
from cls.enumeration import enumerate_terms, interpret_term
from cls.fcl import FiniteCombinatoryLogic

from cls.types import Arrow, Constructor, Literal, Param, Product, SetTo, TVar, Type


def set_plus_one(b: str) -> SetTo:
    def _inner(vars: dict[str, Literal]) -> int:
        return vars[b].value + 1

    return SetTo(_inner)


def labyrinth_set_to() -> None:
    def make_is_free(l_str: list[str]) -> Callable[[dict[str, Literal]], bool]:
        return lambda vars: bool(l_str[vars["b"].value][vars["a"].value] == " ")

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

    free: Callable[[str, str], Type[str]] = lambda a, b: Constructor(
        "free", Product(TVar(a), TVar(b))
    )
    pos: Callable[[str, str], Type[str]] = lambda a, b: Constructor(
        "pos", Product(TVar(a), TVar(b))
    )

    repo: dict[
        Callable[[Any, Any, Any, Any, Any], str] | Callable[[Any, Any], str] | str,
        Param[str] | Type[str],
    ] = {
        FREE: Param(
            "a",
            int,
            lambda _: True,
            Param("b", int, make_is_free(labyrinth_str), free("a", "b")),
        ),
        U: Param(
            "a",
            int,
            lambda _: True,
            Param(
                "b",
                int,
                set_plus_one("a"),
                Param(
                    "c",
                    int,
                    lambda _: True,
                    Arrow(pos("c", "b"), Arrow(free("c", "a"), pos("c", "a"))),
                ),
            ),
        ),
        D: Param(
            "a",
            int,
            lambda _: True,
            Param(
                "b",
                int,
                set_plus_one("a"),
                Param(
                    "c",
                    int,
                    lambda _: True,
                    Arrow(pos("c", "a"), Arrow(free("c", "b"), pos("c", "b"))),
                ),
            ),
        ),
        L: Param(
            "a",
            int,
            lambda _: True,
            Param(
                "b",
                int,
                set_plus_one("a"),
                Param(
                    "c",
                    int,
                    lambda _: True,
                    Arrow(pos("b", "c"), Arrow(free("a", "c"), pos("a", "c"))),
                ),
            ),
        ),
        R: Param(
            "a",
            int,
            lambda _: True,
            Param(
                "b",
                int,
                set_plus_one("a"),
                Param(
                    "c",
                    int,
                    lambda _: True,
                    Arrow(pos("a", "c"), Arrow(free("b", "c"), pos("b", "c"))),
                ),
            ),
        ),
        "START": Constructor("pos", Product(Literal(0, int), Literal(0, int))),
    }

    literals = {int: list(range(10))}

    print("▒▒▒▒▒▒▒▒▒▒▒▒")
    for line in labyrinth_str:
        print(f"▒{line}▒")
    print("▒▒▒▒▒▒▒▒▒▒▒▒")

    fin = Constructor("pos", Product(Literal(9, int), Literal(9, int)))

    fcl: FiniteCombinatoryLogic[
        str, Callable[[Any, Any, Any, Any, Any], str] | Callable[[Any, Any], str] | str
    ] = FiniteCombinatoryLogic(repo, literals=literals)

    grammar = fcl.inhabit(fin)

    for term in enumerate_terms(fin, grammar, 3):
        print(interpret_term(term))


if __name__ == "__main__":
    labyrinth_set_to()
