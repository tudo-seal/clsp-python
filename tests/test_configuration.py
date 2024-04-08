import logging
from typing import TypeVar
import unittest
from clsp import FiniteCombinatoryLogic, enumerate_terms
from clsp.extras import configurable
from clsp.enumeration import interpret_term
from clsp.types import Constructor

T = TypeVar("T")


Int = Constructor("Int")


def add(a: int, b: int) -> int:
    return a + b


def mul(a: int, b: int) -> int:
    return a * b


@configurable
def modulo_add(a: int, b: int, modulo: int) -> int:
    return (a + b) % modulo


@configurable
def modulo_mul(a: int, b: int, modulo: int) -> int:
    return (a * b) % modulo


@configurable
def conf_str_add(a: str, b: str, config: str) -> str:
    return f"({a} + {b} [{config=}])"


@configurable(use_config=False)
def conf_str_mul(a: str, b: str) -> str:
    return f"({a} * {b})"


def str_add(a: str, b: str) -> str:
    return f"({a} + {b})"


def str_mul(a: str, b: str) -> str:
    return f"({a} * {b})"


class TestList(unittest.TestCase):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(module)s %(levelname)s: %(message)s",
        # level=logging.INFO,
    )

    def setUp(self) -> None:
        exampleRepo = {add: Int**Int**Int, mul: Int**Int**Int, 3: Int, 4: Int}
        self.results = enumerate_terms(Int, FiniteCombinatoryLogic(exampleRepo).inhabit(Int), 100)

    def test_str_conf(self) -> None:
        """
        Use config only in combinator mul and 3 and not in add and 4.
        """
        conf_algebra = {add: conf_str_add, mul: conf_str_mul, 3: lambda conf: conf, 4: "4"}

        for z in self.results:
            conf_interpretation = interpret_term(z, conf_algebra)
            self.assertTrue(callable(conf_interpretation) or conf_interpretation == "4")
            if callable(conf_interpretation):
                self.logger.info("%s", conf_interpretation("a"))
                self.logger.info("%s", conf_interpretation("b"))

    def test_mod_conf(self) -> None:
        modulo_algebra = {
            add: modulo_add,
            mul: modulo_mul,
            3: lambda conf: 3 % conf,
            4: lambda conf: 4 % conf,
        }
        for z in self.results:
            mod_interpretation = interpret_term(z, modulo_algebra)
            self.logger.info("%s", mod_interpretation(3))


if __name__ == "__main__":
    unittest.main()
