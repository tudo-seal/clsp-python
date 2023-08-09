import logging
import unittest
from picls.dsl import DSL, Requires
from picls.enumeration import enumerate_terms, interpret_term
from picls.fcl import FiniteCombinatoryLogic
from picls.types import (
    Arrow,
    Intersection,
    Param,
    SetTo,
    Constructor,
    Product,
)


class TestDSL(unittest.TestCase):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(module)s %(levelname)s: %(message)s",
        # level=logging.INFO,
    )

    def setUp(self) -> None:
        self.a = Constructor("a")
        self.b = Constructor("b")
        self.c = Constructor("c")

    def test_arrow(self) -> None:
        self.assertEqual(
            self.a**self.b**self.c, Arrow(self.a, Arrow(self.b, self.c))
        )

    def test_product(self) -> None:
        self.assertEqual(
            self.a * self.b * self.c, Product(Product(self.a, self.b), self.c)
        )

    def test_constructor(self) -> None:
        self.assertEqual("a" @ self.b, Constructor("a", self.b))

    def test_instersection(self) -> None:
        self.assertEqual(
            self.a & self.b & self.c, Intersection(Intersection(self.a, self.b), self.c)
        )

    def test_param(self) -> None:
        def X(x: int, y: int, a: int, z: str) -> str:
            return f"X {x} {y} {a} {z}"

        term1 = (
            DSL()
            .Use("x", "int")
            .With(lambda x: x < 4 and x > 1)
            .Use("y", "int")
            .As(lambda x: x + 1)
            .Use("a", "int")
            .With(lambda y, x: y > x)
            .Use("z", "str")
            .In(self.a)
        )

        grammar1 = FiniteCombinatoryLogic(
            {X: term1}, literals={"int": [1, 2, 3, 4], "str": ["a", "b"]}
        ).inhabit(self.a)

        result1 = {interpret_term(term) for term in enumerate_terms(self.a, grammar1)}

        term2 = Param(
            "x",
            "int",
            lambda vars: bool(vars["x"].value < 4 and vars["x"].value > 1),
            Param(
                "y",
                "int",
                SetTo(lambda vars: vars["x"].value + 1),
                Param(
                    "a",
                    "int",
                    lambda vars: bool(vars["y"].value > vars["x"].value),
                    Param("z", "str", DSL.TRUE, self.a),
                ),
            ),
        )
        grammar2 = FiniteCombinatoryLogic(
            {X: term2}, literals={"int": [1, 2, 3, 4], "str": ["a", "b"]}
        ).inhabit(self.a)

        result2 = {interpret_term(term) for term in enumerate_terms(self.a, grammar2)}

        self.assertEqual(result1, result2)

    def test_req_prov(self) -> None:
        self.assertEqual(
            Requires(self.a, self.b).Provides(self.c),
            Arrow(self.a, Arrow(self.b, self.c)),
        )


if __name__ == "__main__":
    unittest.main()
