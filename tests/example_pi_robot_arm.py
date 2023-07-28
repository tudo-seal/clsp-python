from __future__ import annotations
from picls.enumeration import enumerate_terms, interpret_term
from picls.fcl import FiniteCombinatoryLogic
from picls.types import (
    Arrow,
    Constructor,
    Literal,
    Param,
    TVar,
    Type,
    Intersection,
)


def motorcount() -> None:
    pass


def robotarm() -> None:
    def c(x: Type) -> Type:
        return Constructor("c", x)

    class Part:
        def __init__(self, name: str):
            self.name = name

        def __call__(self, *collect: Part) -> str:
            return (self.name + " params: " + str(collect)).replace("\\", "")

    repo: dict[Part, Type | Param] = {
        Part("motor"): Param(
            "current_motor_count",
            int,
            lambda _: True,
            Param(
                "new_motor_count",
                int,
                lambda vars: bool(
                    vars["current_motor_count"].value + 1
                    == vars["new_motor_count"].value
                ),
                Intersection(
                    Arrow(Constructor("Structural"), Constructor("Motor")),
                    Arrow(c(TVar("current_motor_count")), c(TVar("new_motor_count"))),
                ),
            ),
        ),
        Part("Link"): Param(
            "current_motor_count",
            int,
            lambda _: True,
            Intersection(
                Arrow(Constructor("Motor"), Constructor("Structural")),
                Arrow(c(TVar("current_motor_count")), c(TVar("current_motor_count"))),
            ),
        ),
        Part("ShortLink"): Param(
            "current_motor_count",
            int,
            lambda _: True,
            Intersection(
                Arrow(Constructor("Motor"), Constructor("Structural")),
                Arrow(c(TVar("current_motor_count")), c(TVar("current_motor_count"))),
            ),
        ),
        Part("Effector"): Intersection(Constructor("Structural"), c(Literal(0, int))),
        Part("Base"): Param(
            "current_motor_count",
            int,
            lambda _: True,
            Intersection(
                Arrow(Constructor("Motor"), Constructor("Base")),
                Arrow(c(TVar("current_motor_count")), c(TVar("current_motor_count"))),
            ),
        ),
    }

    literals = {int: list(range(10))}

    fcl: FiniteCombinatoryLogic[Part] = FiniteCombinatoryLogic(repo, literals=literals)
    query = Intersection(Constructor("Base"), c(Literal(3, int)))
    grammar = fcl.inhabit(query)
    # print(grammar.show())

    for term in enumerate_terms(query, grammar):
        print(interpret_term(term))


if __name__ == "__main__":
    robotarm()
