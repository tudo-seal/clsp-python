from typing import Any
from cls.enumeration import enumerate_terms, interpret_term
from cls.fcl import FiniteCombinatoryLogic
from cls.types import Arrow, Constructor, Literal, Param, TVar, Type


def counting() -> None:
    def c(x: Type) -> Type:
        return Constructor("c", x)

    class X:
        def __call__(self, i: int, j: int, y: Any) -> Any:
            return f"X <{i}> <{j}> ({y})"

    repo: dict[X | str, Param | Type] = {
        X(): Param(
            "a",
            int,
            lambda _: True,
            Param(
                "b",
                int,
                lambda vars: bool(vars["a"].value + 1 == vars["b"].value),
                Arrow(c(TVar("a")), c(TVar("b"))),
            ),
        ),
        "Y": c(Literal(3, int)),
    }
    literals = {int: list(range(20))}

    fcl: FiniteCombinatoryLogic[X | str] = FiniteCombinatoryLogic(
        repo, literals=literals
    )
    grammar = fcl.inhabit(c(Literal(5, int)))
    # print(grammar.show())
    for term in enumerate_terms(c(Literal(5, int)), grammar):
        print(interpret_term(term))


if __name__ == "__main__":
    counting()
