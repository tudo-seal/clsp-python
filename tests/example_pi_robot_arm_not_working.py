################################################
# THIS DOES NOT WORK (BUT IT SHOULD IN FUTURE) #
################################################

from typing import Any
from cls.enumeration import enumerate_terms, interpret_term
from cls.fcl import FiniteCombinatoryLogic
from cls.types import (
    Arrow,
    Constructor,
    Literal,
    Param,
    TVar,
    Type,
    Product,
    Intersection,
    TermParamSpec,
)


def motorcount(robot: Any):
    #print("Robot")
    #print(robot.name)
    #print(robot.value)
    #print("RobotEnd")
    return robot.value


def robotarm():
    def c(x: Type[str]) -> Type[str]:
        return Constructor("c", x)

    class Part:
        def __init__(self, name: str, value: int = 0):
            self.name = name
            self.value = value

        def get_additional_value(self):
            return 1 if self.name.startswith("Motor") else 0

        def __call__(self, *other: Any) -> Any:
            return Part(
                self.name + " (" + other[-1].name + ")",
                other[-1].value + self.get_additional_value(),
            )
        
        def __repr__(self) -> str:
            return self.name + "{motorcount=" + str(self.value) + "}"

    repo = {
        Part("Motor"): Arrow(Constructor("Structural"), Constructor("Motor")),
        Part("Link"): Arrow(Constructor("Motor"), Constructor("Structural")),
        Part("ShortLink"): Arrow(Constructor("Motor"), Constructor("Structural")),
        Part("Effector"): Constructor("Structural"),
        Part("Base"): Param(
            "current_motor_count",
            int,
            lambda _: True,
            Param(
                "Robot",
                Constructor("Motor"),
                lambda vars: vars["current_motor_count"].value
                == motorcount(interpret_term(vars["Robot"])),
                Intersection(Constructor("Base"), c(TVar("current_motor_count"))),
            ),
        ),
    }

    literals = {int: list(range(10))}

    fcl: FiniteCombinatoryLogic[str, Any] = FiniteCombinatoryLogic(
        repo, literals=literals
    )
    query = Intersection(Constructor("Base"), c(Literal(3, int)))
    grammar = fcl.inhabit(query)
    # print(grammar.show())

    for term in enumerate_terms(query, grammar):
        print(interpret_term(term))


if __name__ == "__main__":
    robotarm()
