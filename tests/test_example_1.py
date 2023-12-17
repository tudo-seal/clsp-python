from collections.abc import Callable
import unittest
import logging

from clsp import (
    Type,
    Constructor,
    Arrow,
    Intersection,
    FiniteCombinatoryLogic,
    enumerate_terms,
    Subtypes,
)
from clsp.enumeration import interpret_term


class TestExample1(unittest.TestCase):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(module)s %(levelname)s: %(message)s",
        # level=logging.INFO,
    )

    def test_example1(self) -> None:
        a: Type = Constructor("a")
        b: Type = Constructor("b")
        c: Type = Constructor("c")
        d: Type = Constructor("d")

        X: str = "X"
        Y: str = "Y"
        F: Callable[[str], str] = lambda x: f"F({x})"

        repository: dict[str | Callable[[str], str], Type] = dict(
            {
                X: Intersection(Intersection(a, b), d),
                Y: d,
                F: Intersection(Arrow(a, b), Arrow(d, Intersection(a, c))),
            }
        )
        environment: dict[str, set[str]] = dict()
        subtypes = Subtypes(environment)

        target = Intersection(c, b)
        # target = Var(b)

        fcl = FiniteCombinatoryLogic(repository, subtypes)
        result = fcl.inhabit(target)

        enumerated_result = enumerate_terms(target, result)

        for i, real_result in enumerate(enumerated_result):
            term = interpret_term(real_result)
            self.assertEqual("F(X)", term)
            self.assertEqual(i, 0)
            self.logger.info(term)


if __name__ == "__main__":
    unittest.main()
