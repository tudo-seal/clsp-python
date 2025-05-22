import logging
from typing import Any
import unittest
from clsp.dsl import DSL
from clsp.synthesizer import Synthesizer, Specification
from clsp.types import Constructor, Literal, Var
from clsp.inspector import Inspector

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
        componentSpecifications: dict[Part, Specification] = {
            Part("Motor"): Constructor("Structural") ** Constructor("Motor"),
            Part("Link"): Constructor("Motor") ** Constructor("Structural"),
            Part("ShortLink"): Constructor("Motor") ** Constructor("Structural"),
            Part("Effector"): Constructor("Structural"),
            Part("Base"): DSL()
            .Parameter("current_motor_count", "int")
            .Argument("Robot", Constructor("Motor"))
            .Constraint(
                lambda vars: vars["current_motor_count"]
                == motorcount(vars["Robot"].interpret())
            )
            .Suffix(Constructor("Base") & ("c" @ Var("current_motor_count"))),
        }

        parameterSpace = {"int": list(range(10))}

        synthesizer: Synthesizer[Part] = Synthesizer(componentSpecifications, parameterSpace)
        inspector = Inspector()
        inspector.inspect(componentSpecifications, parameterSpace)
        query = Constructor("Base") & ("c" @ Literal(3, "int"))
        grammar = synthesizer.constructSolutionSpace(query)
        self.trees = list(grammar.enumerate_trees(query, 5, 10))
        # self.logger.info(grammar.show())

    def test_count(self) -> None:
        self.assertEqual(4, len(self.trees))

    def test_elements(self) -> None:
        results = [
            "Base (Motor (ShortLink (Motor (ShortLink (Motor (Effector)))))){motorcount=3}",
            "Base (Motor (Link (Motor (Link (Motor (Effector)))))){motorcount=3}",
            "Base (Motor (Link (Motor (ShortLink (Motor (Effector)))))){motorcount=3}",
            "Base (Motor (ShortLink (Motor (Link (Motor (Effector)))))){motorcount=3}",
        ]
        for tree in self.trees:
            self.logger.info(tree.interpret())
            self.assertIn(str(tree.interpret()), results)


if __name__ == "__main__":
    unittest.main()
