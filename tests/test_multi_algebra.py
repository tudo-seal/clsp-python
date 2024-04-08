import logging
from typing import TypeVar
import unittest
from clsp import FiniteCombinatoryLogic, enumerate_terms
from clsp.enumeration import interpret_term
from clsp.types import Constructor

T = TypeVar("T")


Int = Constructor("Int")


def add(a: int, b: int) -> int:
    return a + b


def str_add(a: str, b: str) -> str:
    return f"({a} + {b})"


def mul(a: int, b: int) -> int:
    return a * b


def str_mul(a: str, b: str) -> str:
    return f"({a} * {b})"


class TestList(unittest.TestCase):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(module)s %(levelname)s: %(message)s",
        # level=logging.INFO,
    )

    def setUp(self) -> None:
        exampleRepo = {add: Int**Int**Int, mul: Int**Int**Int, 3: Int, 4: Int, 5: Int}
        self.results = enumerate_terms(Int, FiniteCombinatoryLogic(exampleRepo).inhabit(Int))

    def test_print(self) -> None:
        str_algebra = {add: str_add, mul: str_mul, 3: "3", 4: "4", 5: "5"}

        for z in self.results:
            string_interpretation = interpret_term(z, str_algebra)
            evaluation = eval(string_interpretation)
            cls_eval = interpret_term(z)
            self.logger.info("%s = %s", string_interpretation, cls_eval)
            self.assertEqual(evaluation, cls_eval)


if __name__ == "__main__":
    unittest.main()
