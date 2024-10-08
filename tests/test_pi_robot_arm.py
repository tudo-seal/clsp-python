from __future__ import annotations
import logging
import unittest
from clsp.enumeration import enumerate_terms, interpret_term
from clsp.fcl import FiniteCombinatoryLogic
from clsp.dsl import DSL
from clsp.types import (
    Constructor,
    Literal,
    Param,
    LVar,
    Type,
)


def motorcount() -> None:
    pass


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
            Part("motor"): DSL()
            .Use("current_motor_count", "int")
            .Use("new_motor_count", "int")
            .AsRaw(lambda vars: vars["current_motor_count"] + 1)
            .In(
                Constructor("Structural") ** Constructor("Motor")
                & ("c" @ LVar("current_motor_count")) ** ("c" @ LVar("new_motor_count"))
            ),
            Part("Link"): DSL()
            .Use("current_motor_count", "int")
            .In(
                Constructor("Motor") ** Constructor("Structural")
                & (("c" @ LVar("current_motor_count")) ** ("c" @ LVar("current_motor_count")))
            ),
            Part("ShortLink"): DSL()
            .Use("current_motor_count", "int")
            .In(
                Constructor("Motor") ** Constructor("Structural")
                & (("c" @ LVar("current_motor_count")) ** ("c" @ LVar("current_motor_count")))
            ),
            Part("Effector"): Constructor("Structural") & ("c" @ Literal(0, "int")),
            Part("Base"): DSL()
            .Use("current_motor_count", "int")
            .In(
                Constructor("Motor") ** Constructor("Base")
                & (("c" @ LVar("current_motor_count")) ** ("c" @ LVar("current_motor_count")))
            ),
        }

        literals = {"int": list(range(10))}

        fcl: FiniteCombinatoryLogic[Part] = FiniteCombinatoryLogic(repo, literals=literals)
        query = Constructor("Base") & ("c" @ (Literal(3, "int")))
        grammar = fcl.inhabit(query)
        self.terms = list(enumerate_terms(query, grammar))
        # self.logger.info(grammar.show())

    def test_count(self) -> None:
        self.assertEqual(4, len(self.terms))

    def test_elements(self) -> None:
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
