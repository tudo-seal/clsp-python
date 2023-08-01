################################################
# THIS DOES NOT WORK (BUT IT SHOULD IN FUTURE) #
################################################

import logging
from typing import Any
import unittest
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


def motorcount(robot: Any) -> Any:
    # self.logger.info("Robot")
    # self.logger.info(robot.name)
    # self.logger.info(robot.value)
    # self.logger.info("RobotEnd")
    return robot.value


def c(x: Type) -> Type:
    return Constructor("c", x)


class Part:
    def __init__(self, name: str, value: int = 0):
        self.name = name
        self.value = value

    def get_additional_value(self) -> int:
        return 1 if self.name.startswith("Motor") else 0

    def __call__(self, *other: Any) -> Any:
        return Part(
            self.name + " (" + other[-1].name + ")",
            other[-1].value + self.get_additional_value(),
        )

    def __repr__(self) -> str:
        return self.name + "{motorcount=" + str(self.value) + "}"


class TestRobotArm(unittest.TestCase):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(module)s %(levelname)s: %(message)s",
        # level=logging.INFO,
    )

    def setUp(self) -> None:
        repo: dict[Part, Param | Type] = {
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
                    lambda vars: bool(
                        vars["current_motor_count"].value
                        == motorcount(interpret_term(vars["Robot"]))
                    ),
                    Intersection(Constructor("Base"), c(TVar("current_motor_count"))),
                ),
            ),
        }

        literals = {int: list(range(10))}

        fcl: FiniteCombinatoryLogic[Part] = FiniteCombinatoryLogic(
            repo, literals=literals
        )
        query = Intersection(Constructor("Base"), c(Literal(3, int)))
        grammar = fcl.inhabit(query)
        self.terms = list(enumerate_terms(query, grammar))
        # self.logger.info(grammar.show())

    def test_count(self):
        self.assertEqual(4, len(self.terms))

    def test_elements(self):
        results = [
            "Base (Motor (ShortLink (Motor (ShortLink (Motor (Effector)))))){motorcount=3}",
            "Base (Motor (Link (Motor (Link (Motor (Effector)))))){motorcount=3}",
            "Base (Motor (Link (Motor (ShortLink (Motor (Effector)))))){motorcount=3}",
            "Base (Motor (ShortLink (Motor (Link (Motor (Effector)))))){motorcount=3}",
        ]
        for term in self.terms:
            self.logger.info(interpret_term(term))
            self.assertIn(str(interpret_term(term)), results)


if __name__ == "__main__":
    unittest.main()
