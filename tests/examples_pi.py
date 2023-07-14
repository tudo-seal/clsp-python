from collections.abc import Callable
from typing import Any
from cls import FiniteCombinatoryLogic
from cls.enumeration import enumerate_terms, interpret_term
from cls.types import Arrow, Constructor, Literal, Param, Product, TVar, Type


def counting():
    def c(x: Type[str]) -> Type[str]:
        return Constructor("c", x)

    class X:
        def __call__(self, i: int, j: int, y: Any) -> Any:
            return f"X <{i}> <{j}> ({y})"

    repo = {
        X(): Param(
            "a",
            int,
            lambda _: True,
            Param(
                "b",
                int,
                lambda vars: vars["a"].name + 1 == vars["b"].name,
                Arrow(c(TVar("a")), c(TVar("b"))),
            ),
        ),
        "Y": c(Literal(3, int)),
    }
    literals = {int: list(range(20))}

    fcl: FiniteCombinatoryLogic[str, Any] = FiniteCombinatoryLogic(
        repo, literals=literals
    )
    grammar = fcl.inhabit(c(Literal(5, int)))
    print(grammar.show())
    for term in enumerate_terms(c(Literal(5, int)), grammar):
        print(interpret_term(term))


def labyrinth():
    def make_is_free(l_str: list[str]) -> Callable[[dict[str, Literal]], bool]:
        return lambda vars: l_str[vars["b"].name][vars["a"].name] == "."

    labyrinth_str = [
        ".x........",
        ".x........",
        ".xxxxxxxxx",
        ".........x",
        ".xxxxxxx.x",
        ".x.....x.x",
        ".x.....x..",
        ".x.....xx.",
        ".xxxxxxxx.",
        "..........",
    ]

    FREE = lambda a, b: f"FREE({a}, {b})"
    U = lambda a, b, c, p, f: f"{p} => UP({c}, {a})"
    D = lambda a, b, c, p, f: f"{p} => DOWN({c}, {b})"
    L = lambda a, b, c, p, f: f"{p} => LEFT({a}, {c})"
    R = lambda a, b, c, p, f: f"{p} => RIGHT({b}, {c})"

    free = lambda a, b: Constructor("free", Product(TVar(a), TVar(b)))
    pos = lambda a, b: Constructor("pos", Product(TVar(a), TVar(b)))

    repo = {
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
                lambda vars: vars["b"].name == vars["a"].name + 1,
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
                lambda vars: vars["b"].name == vars["a"].name + 1,
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
                lambda vars: vars["b"].name == vars["a"].name + 1,
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
                lambda vars: vars["b"].name == vars["a"].name + 1,
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

    fcl: FiniteCombinatoryLogic[Any, Any] = FiniteCombinatoryLogic(
        repo, literals=literals
    )
    fin = Constructor("pos", Product(Literal(9, int), Literal(9, int)))
    grammar = fcl.inhabit(fin)
    # print(grammar.show())
    for line in labyrinth_str:
        print(line)
    for term in enumerate_terms(fin, grammar, 2):
        print(interpret_term(term))


if __name__ == "__main__":
    print("Counting Example:")
    counting()
    print("\n\nLabyrinth Example:")
    labyrinth()
