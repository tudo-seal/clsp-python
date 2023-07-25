import timeit

from collections.abc import Callable
from typing import Any
from cls.enumeration import enumerate_terms, interpret_term
from cls.fcl import FiniteCombinatoryLogic

from cls.types import Arrow, Constructor, Literal, Param, Product, SetTo, TVar, Type


def set_plus_one(b: str) -> SetTo:
    def _inner(vars: dict[str, Literal]) -> int:
        return int(vars[b].value + 1)

    return SetTo(_inner)


# pseudo-random labyrinth
def is_free(row: int, col: int) -> bool:
    SEED = 0
    if row == col:
        return True
    else:
        return (
            pow(11, (row + col + SEED) * (row + col + SEED) + col + 7, 1000003) % 5 > 0
        )


def int_to_type(x: int) -> Type[str]:
    return Constructor(str(x))


def free(row: int, col: int) -> Type[str]:
    return Constructor("Free", Product(int_to_type(row), int_to_type(col)))


def pos(row: int, col: int) -> Type[str]:
    return Constructor("Pos", Product(int_to_type(row), int_to_type(col)))


def move(
    drow_from: int, dcol_from: int, drow_to: int, dcol_to: int, SIZE: int
) -> Type[str]:
    return Type.intersect(
        [
            Arrow(
                pos(row + drow_from, col + dcol_from),
                Arrow(
                    free(row + drow_to, col + dcol_to),
                    pos(row + drow_to, col + dcol_to),
                ),
            )
            for row in range(0, SIZE)
            for col in range(0, SIZE)
        ]
    )


def labyrinth(SIZE: int = 10, output: bool = True) -> float:
    if output:
        for row in range(SIZE):
            for col in range(SIZE):
                if is_free(row, col):
                    print("-", end="")
                else:
                    print("#", end="")
            print("")

    free: Callable[[str, str], Type[str]] = lambda a, b: Constructor(
        "free", Product(TVar(a), TVar(b))
    )
    pos: Callable[[str, str], Type[str]] = lambda a, b: Constructor(
        "pos", Product(TVar(a), TVar(b))
    )

    FREE = lambda a, b: f"FREE({a}, {b})"
    U = lambda a, b, c, p, f: f"{p} => UP({c}, {a})"
    D = lambda a, b, c, p, f: f"{p} => DOWN({c}, {b})"
    L = lambda a, b, c, p, f: f"{p} => LEFT({a}, {c})"
    R = lambda a, b, c, p, f: f"{p} => RIGHT({b}, {c})"

    repo: dict[
        Callable[[Any, Any, Any, Any, Any], str] | Callable[[Any, Any], str] | str,
        Param[str] | Type[str],
    ] = {
        FREE: Param(
            "a",
            int,
            lambda _: True,
            Param(
                "b",
                int,
                lambda vars: is_free(vars["b"].value, vars["a"].value),
                free("a", "b"),
            ),
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
                    Param(
                        "p",
                        pos("c", "b"),
                        lambda _: True,
                        Param("f", free("c", "a"), lambda _: True, pos("c", "a")),
                    ),
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
                    Param(
                        "p",
                        pos("c", "a"),
                        lambda _: True,
                        Param("f", free("c", "b"), lambda _: True, pos("c", "b")),
                    ),
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
                    Param(
                        "p",
                        pos("b", "c"),
                        lambda _: True,
                        Param("f", free("a", "c"), lambda _: True, pos("a", "c")),
                    ),
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
                    Param(
                        "p",
                        pos("a", "c"),
                        lambda _: True,
                        Param("f", free("b", "c"), lambda _: True, pos("b", "c")),
                    ),
                ),
            ),
        ),
        "START": Constructor("pos", Product(Literal(0, int), Literal(0, int))),
    }

    start = timeit.default_timer()

    literals = {int: list(range(SIZE))}
    target = Constructor("pos", Product(Literal(SIZE - 1, int), Literal(SIZE - 1, int)))

    fcl: FiniteCombinatoryLogic[
        str, Callable[[Any, Any, Any, Any, Any], str] | Callable[[Any, Any], str] | str
    ] = FiniteCombinatoryLogic(repo, literals=literals)

    if output:
        print("Time (Constructor): ", timeit.default_timer() - start)
    start = timeit.default_timer()

    grammar = fcl.inhabit(target)
    itime = timeit.default_timer() - start
    if output:
        print(f"Time (Inhabitation): {itime}")

    start = timeit.default_timer()
    for term in enumerate_terms(target, grammar, 3):
        t = interpret_term(term)
        if output:
            print(t)
    etime = timeit.default_timer() - start
    if output:
        print(f"Time (Enumerate/Interpret): {etime}")
    return itime + etime


if __name__ == "__main__":
    labyrinth()
