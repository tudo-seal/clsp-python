import logging
import unittest
from picls import (
    Type,
    Constructor,
    Arrow,
)
from picls.enumeration import enumerate_terms, interpret_term
from picls.fcl import FiniteCombinatoryLogic
from picls.types import Param, TVar


class TestLiterals(unittest.TestCase):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(module)s %(levelname)s: %(message)s",
        # level=logging.INFO,
    )

    def test_liteals(self) -> None:
        # a: Type = Constructor("a")
        # b: Type = Constructor("b")
        c: Type = Constructor("c")
        # d: Type = Literal(3, int)

        # X: str = "X"
        Y = lambda x, y: f"Y {x} {y}"
        # F: Callable[[str], str] = lambda x: f"F({x})"

        repository = dict({Y: Param("x", int, lambda _: True, Arrow(TVar("x"), c))})
        # for real_result in inhabit_and_interpret(repository, [Literal("3", int)]):
        #     self.logger.info(real_result)
        result = FiniteCombinatoryLogic(repository, literals={int: [3]}).inhabit(c)
        self.logger.info(result.show())
        for term in enumerate_terms(c, result):
            self.logger.info(interpret_term(term))
            self.assertEqual("Y 3 3", interpret_term(term))


if __name__ == "__main__":
    unittest.main()
