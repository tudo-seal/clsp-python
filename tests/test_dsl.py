import logging
import unittest
from clsp.dsl import DSL, Requires
from clsp.enumeration import enumerate_terms, interpret_term
from clsp.fcl import FiniteCombinatoryLogic
from clsp.types import (
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
        self.assertEqual(self.a**self.b**self.c, Arrow(self.a, Arrow(self.b, self.c)))

    def test_product(self) -> None:
        self.assertEqual(self.a * self.b * self.c, Product(Product(self.a, self.b), self.c))

    def test_constructor(self) -> None:
        self.assertEqual("a" @ self.b, Constructor("a", self.b))

    def test_instersection(self) -> None:
        self.assertEqual(
            self.a & self.b & self.c, Intersection(Intersection(self.a, self.b), self.c)
        )

    def test_param(self) -> None:
        def X(x: int, y: int, a: int, z: str, b: int) -> str:
            return f"X {x} {y} {a} {z} {b}"

        term1 = (
            DSL()
            .Use("x", "int")
            .With(lambda x: x < 4 and x > 1)
            .Use("y", "int")
            .As(lambda x: x + 1)
            .Use("a", "int")
            .With(lambda y, x: y > x)
            .Use("z", "str")
            .Use("b", "int")
            .As(lambda: 3)
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
                    Param(
                        "z",
                        "str",
                        DSL.TRUE,
                        Param("b", "int", SetTo(lambda vars: 3), self.a),
                    ),
                ),
            ),
        )
        grammar2 = FiniteCombinatoryLogic(
            {X: term2}, literals={"int": [1, 2, 3, 4], "str": ["a", "b"]}
        ).inhabit(self.a)

        result2 = {interpret_term(term) for term in enumerate_terms(self.a, grammar2)}

        result3 = set()
        for x in [1, 2, 3, 4]:
            if not (x < 4 and x > 1):
                continue
            y = x + 1
            for a in [1, 2, 3, 4]:
                if not y > x:
                    continue
                for z in ["a", "b"]:
                    b = 3
                    result3.add(f"X {x} {y} {a} {z} {b}")

        self.assertEqual(result1, result2)
        self.assertEqual(result1, result3)

    def test_req_prov(self) -> None:
        self.assertEqual(
            Requires(self.a, self.b).Provides(self.c),
            Arrow(self.a, Arrow(self.b, self.c)),
        )


if __name__ == "__main__":
    unittest.main()
