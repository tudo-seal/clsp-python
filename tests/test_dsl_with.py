# demonstration of predicates on literals using the "With" method

import logging
import unittest
from clsp.dsl import DSL
from clsp.synthesizer import Synthesizer
from clsp.types import Constructor

class TestDSLWith(unittest.TestCase):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(module)s %(levelname)s: %(message)s",
        # level=logging.INFO,
    )

    def setUp(self) -> None:
        self.a = Constructor("a")

    def test_param(self) -> None:
        def X(x: int, y: int) -> str:
            return f"X {x} {y}"

        specification1 = (
            DSL()
            .Use("x", "int")
            .With(lambda vars: vars["x"] > 1)
            .With(lambda vars: vars["x"] < 4)
            .Use("y", "int", lambda vars: [vars["x"] + 1])
            .In(self.a)
        )

        specification2 = (
            DSL()
            .Use("x", "int")
            .With(lambda vars: vars["x"] < 4 and vars["x"] > 1)
            .Use("y", "int", lambda vars: [vars["x"] + 1])
            .In(self.a)
        )

        specification3 = (
            DSL()
            .Use("x", "int")
            .Use("y", "int", lambda vars: [vars["x"] + 1])
            .With(lambda vars: vars["x"] < 4 and vars["x"] > 1)
            .In(self.a)
        )

        grammar1 = Synthesizer(
            {X: specification1}, parameterSpace={"int": [1, 2, 3, 4], "str": ["a", "b"]}
        ).constructSolutionSpace(self.a)

        result1 = {tree.interpret() for tree in grammar1.enumerate_trees(self.a)}

        grammar2 = Synthesizer(
            {X: specification2}, parameterSpace={"int": [1, 2, 3, 4], "str": ["a", "b"]}
        ).constructSolutionSpace(self.a)

        result2 = {tree.interpret() for tree in grammar2.enumerate_trees(self.a)}

        grammar3 = Synthesizer(
            {X: specification3}, parameterSpace={"int": [1, 2, 3, 4], "str": ["a", "b"]}
        ).constructSolutionSpace(self.a)

        result3 = {tree.interpret() for tree in grammar3.enumerate_trees(self.a)}

        expectedResult = set()
        for x in [1, 2, 3, 4]:
            if not (x < 4 and x > 1):
                continue
            y = x + 1
            if y not in [1, 2, 3, 4]:
                continue
            expectedResult.add(f"X {x} {y}")

        self.assertEqual(expectedResult, result1)
        self.assertEqual(expectedResult, result2)
        self.assertEqual(expectedResult, result3)

if __name__ == "__main__":
    unittest.main()
