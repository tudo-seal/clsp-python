from __future__ import annotations
import logging
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


def motorcount() -> None:
    pass


def c(x: Type) -> Type:
    return Constructor("c", x)


class Part:
    def __init__(self, name: str):
        self.name = name

    def __call__(self, *collect: Part) -> str:
        return (self.name + " params: " + str(collect)).replace("\\", "")

    def __repr__(self) -> str:
        return self.name


class TestRobotArm(unittest.TestCase):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(module)s %(levelname)s: %(message)s",
        # level=logging.INFO,
    )

    def setUp(self) -> None:
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
                        Arrow(
                            c(TVar("current_motor_count")), c(TVar("new_motor_count"))
                        ),
                    ),
                ),
            ),
            Part("Link"): Param(
                "current_motor_count",
                int,
                lambda _: True,
                Intersection(
                    Arrow(Constructor("Motor"), Constructor("Structural")),
                    Arrow(
                        c(TVar("current_motor_count")), c(TVar("current_motor_count"))
                    ),
                ),
            ),
            Part("ShortLink"): Param(
                "current_motor_count",
                int,
                lambda _: True,
                Intersection(
                    Arrow(Constructor("Motor"), Constructor("Structural")),
                    Arrow(
                        c(TVar("current_motor_count")), c(TVar("current_motor_count"))
                    ),
                ),
            ),
            Part("Effector"): Intersection(
                Constructor("Structural"), c(Literal(0, int))
            ),
            Part("Base"): Param(
                "current_motor_count",
                int,
                lambda _: True,
                Intersection(
                    Arrow(Constructor("Motor"), Constructor("Base")),
                    Arrow(
                        c(TVar("current_motor_count")), c(TVar("current_motor_count"))
                    ),
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
            "Base params: (3, 'motor params: (2, 3, 'Link params: (2, 'motor params: (1, 2, "
            "\"ShortLink params: (1, 'motor params: (0, 1, Effector)')\")')')')",
            "Base params: (3, 'motor params: (2, 3, 'ShortLink params: (2, 'motor params: (1, 2, "
            "\"Link params: (1, 'motor params: (0, 1, Effector)')\")')')')",
            "Base params: (3, 'motor params: (2, 3, 'ShortLink params: (2, 'motor params: (1, 2, "
            "\"ShortLink params: (1, 'motor params: (0, 1, Effector)')\")')')')",
            "Base params: (3, 'motor params: (2, 3, 'Link params: (2, 'motor params: (1, 2, "
            "\"Link params: (1, 'motor params: (0, 1, Effector)')\")')')')",
        ]
        for term in self.terms:
            self.logger.info(interpret_term(term))
            self.assertIn(interpret_term(term), results)


if __name__ == "__main__":
    unittest.main()
