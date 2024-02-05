import logging
import unittest
from clsp.dsl import DSL
from clsp.enumeration import enumerate_terms, interpret_term
from clsp.fcl import FiniteCombinatoryLogic
from clsp.types import (
    Constructor,
)


class TestDSL(unittest.TestCase):
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

        term1 = (
            DSL()
            .Use("x", "int")
            .With(lambda x: x > 1)
            .With(lambda x: x < 4)
            .Use("y", "int")
            .As(lambda x: x + 1)
            .In(self.a)
        )

        term2 = (
            DSL()
            .Use("x", "int")
            .With(lambda x: x < 4 and x > 1)
            .Use("y", "int")
            .As(lambda x: x + 1)
            .In(self.a)
        )

        term3 = (
            DSL()
            .Use("x", "int")
            .Use("y", "int")
            .As(lambda x: x + 1)
            .With(lambda x: x < 4 and x > 1)
            .In(self.a)
        )

        term4 = (
            DSL()
            .Use("x", "int")
            .Use("y", "int")
            .With(lambda x: x < 4 and x > 1)
            .As(lambda x: x + 1)
            .In(self.a)
        )

        grammar1 = FiniteCombinatoryLogic(
            {X: term1}, literals={"int": [1, 2, 3, 4], "str": ["a", "b"]}
        ).inhabit(self.a)

        result1 = {interpret_term(term) for term in enumerate_terms(self.a, grammar1)}

        grammar2 = FiniteCombinatoryLogic(
            {X: term2}, literals={"int": [1, 2, 3, 4], "str": ["a", "b"]}
        ).inhabit(self.a)

        result2 = {interpret_term(term) for term in enumerate_terms(self.a, grammar2)}

        grammar3 = FiniteCombinatoryLogic(
            {X: term3}, literals={"int": [1, 2, 3, 4], "str": ["a", "b"]}
        ).inhabit(self.a)

        result3 = {interpret_term(term) for term in enumerate_terms(self.a, grammar3)}

        grammar4 = FiniteCombinatoryLogic(
            {X: term4}, literals={"int": [1, 2, 3, 4], "str": ["a", "b"]}
        ).inhabit(self.a)

        result4 = {interpret_term(term) for term in enumerate_terms(self.a, grammar4)}

        result5 = set()
        for x in [1, 2, 3, 4]:
            if not (x < 4 and x > 1):
                continue
            y = x + 1
            if y not in [1, 2, 3, 4]:
                continue
            result5.add(f"X {x} {y}")

        self.assertEqual(result5, result1)
        self.assertEqual(result5, result2)
        self.assertEqual(result5, result3)
        self.assertEqual(result5, result4)


if __name__ == "__main__":
    unittest.main()
