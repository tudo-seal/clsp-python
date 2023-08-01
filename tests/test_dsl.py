import logging
import unittest
from picls.dsl import TRUE, Use, Requires
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
        p = lambda _: False
        s = lambda _: 3
        term1 = Use("x", int).With(p).Use("y", str).As(s).Use("z", str).In(self.a)
        term2 = Param(
            "x", int, p, Param("y", str, SetTo(s), Param("z", str, TRUE, self.a))
        )
        self.assertEqual(term1, term2)

    def test_req_prov(self) -> None:
        self.assertEqual(
            Requires(self.a, self.b).Provides(self.c),
            Arrow(self.a, Arrow(self.b, self.c)),
        )


if __name__ == "__main__":
    unittest.main()
