from __future__ import annotations
import logging
import unittest
from clsp.synthesizer import Synthesizer, Specification, Contains
from clsp.dsl import DSL
from clsp.types import (
    Constructor,
    Literal,
    Var,
)

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
        componentSpecifications: dict[Part, Specification] = {
            Part("motor"): DSL()
            .Parameter("new_motor_count", "int")
            .Parameter("current_motor_count", "int", lambda vars: [vars["new_motor_count"] - 1])
            .Suffix(
                Constructor("Structural") ** Constructor("Motor")
                & ("c" @ Var("current_motor_count")) ** ("c" @ Var("new_motor_count"))
            ),
            Part("Link"): DSL()
            .Parameter("current_motor_count", "int")
            .Suffix(
                Constructor("Motor") ** Constructor("Structural")
                & (("c" @ Var("current_motor_count")) ** ("c" @ Var("current_motor_count")))
            ),
            Part("ShortLink"): DSL()
            .Parameter("current_motor_count", "int")
            .Suffix(
                Constructor("Motor") ** Constructor("Structural")
                & (("c" @ Var("current_motor_count")) ** ("c" @ Var("current_motor_count")))
            ),
            Part("Effector"): Constructor("Structural") & ("c" @ Literal(0, "int")),
            Part("Base"): DSL()
            .Parameter("current_motor_count", "int")
            .Suffix(
                Constructor("Motor") ** Constructor("Base")
                & (("c" @ Var("current_motor_count")) ** ("c" @ Var("current_motor_count")))
            ),
        }

        class Int(Contains):
            # represents the set of (arbitrary large) natural numbers
            def __contains__(self, value: object) -> bool:
                return isinstance(value, int) and value >= 0

        parameterSpace = {"int": Int()}

        synthesizer: Synthesizer[Part] = Synthesizer(componentSpecifications, parameterSpace)
        query = Constructor("Base") & ("c" @ (Literal(3, "int")))
        grammar = synthesizer.constructSolutionSpace(query)
        self.trees = list(grammar.enumerate_trees(query))
        # self.logger.info(grammar.show())

    def test_count(self) -> None:
        self.assertEqual(4, len(self.trees))

    def test_elements(self) -> None:
        results = [
            "Base params: (3, 'motor params: (3, 2, 'Link params: (2, 'motor params: (2, 1, "
            "\"ShortLink params: (1, 'motor params: (1, 0, Effector)')\")')')')",
            "Base params: (3, 'motor params: (3, 2, 'ShortLink params: (2, 'motor params: (2, 1, "
            "\"Link params: (1, 'motor params: (1, 0, Effector)')\")')')')",
            "Base params: (3, 'motor params: (3, 2, 'ShortLink params: (2, 'motor params: (2, 1, "
            "\"ShortLink params: (1, 'motor params: (1, 0, Effector)')\")')')')",
            "Base params: (3, 'motor params: (3, 2, 'Link params: (2, 'motor params: (2, 1, "
            "\"Link params: (1, 'motor params: (1, 0, Effector)')\")')')')",
        ]
        for tree in self.trees:
            self.logger.info(tree.interpret())
            self.assertIn(tree.interpret(), results)


if __name__ == "__main__":
    unittest.main()
