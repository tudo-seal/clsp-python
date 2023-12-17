import logging
from typing import Any
import unittest
from clsp.enumeration import enumerate_terms, interpret_term
from clsp.fcl import FiniteCombinatoryLogic
from clsp.types import Arrow, Constructor, Literal, Param, LVar, Type


class TestConting(unittest.TestCase):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(module)s %(levelname)s: %(message)s",
        # level=logging.INFO,
    )

    def test_counting(self) -> None:
        def c(x: Type) -> Type:
            return Constructor("c", x)

        class X:
            def __call__(self, i: int, j: int, y: Any) -> Any:
                return f"X <{i}> <{j}> ({y})"

        repo: dict[X | str, Param | Type] = {
            X(): Param(
                "a",
                "int",
                lambda _: True,
                Param(
                    "b",
                    "int",
                    lambda vars: bool(vars["a"].value + 1 == vars["b"].value),
                    Arrow(c(LVar("a")), c(LVar("b"))),
                ),
            ),
            "Y": c(Literal(3, "int")),
        }
        literals = {"int": list(range(20))}

        fcl: FiniteCombinatoryLogic[X | str] = FiniteCombinatoryLogic(
            repo, literals=literals
        )
        grammar = fcl.inhabit(c(Literal(5, "int")))
        # self.logger.info(grammar.show())
        for term in enumerate_terms(c(Literal(5, "int")), grammar):
            self.logger.info(interpret_term(term))
            self.assertEqual("X <4> <5> (X <3> <4> (Y))", interpret_term(term))


if __name__ == "__main__":
    unittest.main()
