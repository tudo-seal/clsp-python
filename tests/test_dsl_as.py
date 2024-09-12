import logging
import unittest
from clsp.dsl import DSL
from clsp.enumeration import enumerate_terms, interpret_term
from clsp.fcl import FiniteCombinatoryLogic
from clsp.types import Type, Constructor, LVar, Literal


class TestDSLAs(unittest.TestCase):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(module)s %(levelname)s: %(message)s",
        # level=logging.INFO,
    )

    def test_param(self) -> None:
        def X(x: bool, y: bool, z: bool) -> str:
            return f"X {x} {y} {z}"

        Gamma = {
            X: DSL()
            .Use("x", "bool")
            .Use("y", "bool")
            .As(lambda x: True)
            .Use("z", "bool")
            .As(lambda x: x)
            .In(
                Constructor("a", LVar("x"))
                & Constructor("b", LVar("y"))
                & Constructor("c", LVar("z"))
            )
        }

        def xyz(x: bool, y: bool, z: bool) -> Type:
            return (
                Constructor("a", Literal(x, "bool"))
                & Constructor("b", Literal(y, "bool"))
                & Constructor("c", Literal(z, "bool"))
            )

        fcl = FiniteCombinatoryLogic(Gamma, literals={"bool": [True, False]})

        for x in [True, False]:
            for y in [True, False]:
                for z in [True, False]:
                    target = xyz(x, y, z)
                    grammar = fcl.inhabit(target)
                    result = {interpret_term(term) for term in enumerate_terms(target, grammar)}
                    self.assertLessEqual(len(result), 1)
                    self.assertTrue(result.issubset({"X True True True", "X False True False"}))

    def test_multi_as1(self) -> None:
        def X(a: int, b: int) -> str:
            return f"X {a} {b}"

        literals = {"int": [0, 1, 2, 3]}
        Gamma = {
            X: DSL(cache=True)
            .Use("a", "int")
            .Use("b", "int")
            .As(lambda a: {a - 1, a + 1}, multi_value=True)
            .In(Constructor("c", LVar("a")))
        }

        fcl = FiniteCombinatoryLogic(Gamma, literals=literals)
        target = Constructor("c", Literal(0, "int"))

        result = fcl.inhabit(target)
        self.assertEqual(
            list(interpret_term(x) for x in enumerate_terms(target, result)), ["X 0 1"]
        )

    def test_multi_as2(self) -> None:
        def X(a: int, b: int) -> str:
            return f"X {a} {b}"

        literals = {"int": [0, 1, 2, 3]}
        Gamma = {
            X: DSL(cache=True)
            .Use("a", "int")
            .Use("b", "int")
            .As(lambda a: {a - 1, a + 1}, multi_value=True)
            .In(Constructor("c", LVar("a")))
        }

        fcl = FiniteCombinatoryLogic(Gamma, literals=literals)
        target = Constructor("c", Literal(1, "int"))

        result = fcl.inhabit(target)
        self.assertSetEqual(
            set(interpret_term(x) for x in enumerate_terms(target, result)), {"X 1 2", "X 1 0"}
        )

    def test_multi_as3(self) -> None:
        def X(a: int, b: int) -> str:
            return f"X {a} {b}"

        literals = {"int": [0, 1, 2, 3]}
        Gamma = {
            X: DSL(cache=True)
            .Use("a", "int")
            .Use("b", "int")
            .As(lambda a: {a - 1, a + 1}, multi_value=True, override=True)
            .In(Constructor("c", LVar("a")))
        }

        fcl = FiniteCombinatoryLogic(Gamma, literals=literals)
        target = Constructor("c", Literal(0, "int"))

        result = fcl.inhabit(target)
        self.assertSetEqual(
            set(interpret_term(x) for x in enumerate_terms(target, result)), {"X 0 -1", "X 0 1"}
        )


if __name__ == "__main__":
    unittest.main()
