import logging
import unittest
from clsp import (
    Type,
    Constructor,
)
from clsp.dsl import DSL
from clsp.enumeration import enumerate_terms, interpret_term
from clsp.fcl import FiniteCombinatoryLogic
from clsp.types import LVar, Literal


class TestLiterals(unittest.TestCase):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(module)s %(levelname)s: %(message)s",
        # level=logging.INFO,
    )

    def test_liteals(self) -> None:
        c: Type = Constructor("c")
        Y = lambda x, y: f"Y {x} {y}"

        repository = {
            Y: DSL().Use("x", "int").In(("d" @ LVar("x")) ** c),
            "3": "d" @ Literal(3, "int"),
        }

        result = FiniteCombinatoryLogic(repository, literals={"int": [3]}).inhabit(c)
        self.logger.info(result.show())
        results = [interpret_term(term) for term in enumerate_terms(c, result)]
        self.logger.info(results)
        self.assertEqual(["Y 3 3"], results)


if __name__ == "__main__":
    unittest.main()
