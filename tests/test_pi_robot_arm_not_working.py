################################################
# THIS DOES NOT WORK (BUT IT SHOULD IN FUTURE) #
################################################

import logging
from typing import Any
import unittest
from picls.dsl import DSL
from picls.enumeration import enumerate_terms, interpret_term
from picls.fcl import FiniteCombinatoryLogic
from picls.types import (
    Constructor,
    Literal,
    Param,
    TVar,
    Type,
)


def motorcount(robot: Any) -> Any:
    # self.logger.info("Robot")
    # self.logger.info(robot.name)
    # self.logger.info(robot.value)
    # self.logger.info("RobotEnd")
    return robot.value


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
            Part("Motor"): Constructor("Structural") ** Constructor("Motor"),
            Part("Link"): Constructor("Motor") ** Constructor("Structural"),
            Part("ShortLink"): Constructor("Motor") ** Constructor("Structural"),
            Part("Effector"): Constructor("Structural"),
            Part("Base"): DSL()
            .Use("current_motor_count", int)
            .Use("Robot", Constructor("Motor"))
            .With(
                lambda current_motor_count, Robot: current_motor_count
                == motorcount(interpret_term(Robot))
            )
            .In(Constructor("Base") & ("c" @ TVar("current_motor_count"))),
        }

        literals = {int: list(range(10))}

        fcl: FiniteCombinatoryLogic[Part] = FiniteCombinatoryLogic(
            repo, literals=literals
        )
        query = Constructor("Base") & ("c" @ Literal(3, int))
        grammar = fcl.inhabit(query)
        self.terms = list(enumerate_terms(query, grammar))
        # self.logger.info(grammar.show())

    def test_count(self) -> None:
        self.assertEqual(4, len(self.terms))

    def test_elements(self) -> None:
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
