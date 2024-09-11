import logging
import unittest
from clsp.dsl import DSL
from clsp.enumeration import enumerate_terms, interpret_term
from clsp.fcl import FiniteCombinatoryLogic
from clsp.types import (
    Type, Constructor, LVar, Literal
)

class TestDSLAs(unittest.TestCase):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(module)s %(levelname)s: %(message)s",
        # level=logging.INFO,
    )

    def test_param(self) -> None:
        def X(x: bool, y: bool, z: bool) -> str:
            return f"X {x} {y} {z}"

        Gamma = {X: DSL()
            .Use("x", "bool")
            .Use("y", "bool")
            .As(lambda x: True)
            .Use("z", "bool")
            .As(lambda x: x)
            .In(Constructor("a", LVar('x')) & Constructor("b", LVar('y')) & Constructor("c", LVar('z')))}

        def xyz(x: bool, y: bool, z: bool) -> Type:
            return Constructor("a", Literal(x, "bool")) & Constructor("b", Literal(y, "bool")) & Constructor("c", Literal(z, "bool"))
        
        fcl = FiniteCombinatoryLogic(Gamma, literals={"bool": [True, False]})

        for x in [True, False]:
            for y in [True, False]:
                for z in [True, False]:
                    target = xyz(x, y, z)
                    grammar = fcl.inhabit(target)
                    result = {interpret_term(term) for term in enumerate_terms(target, grammar)}
                    self.assertLessEqual(len(result), 1)
                    self.assertTrue(result.issubset({"X True True True", "X False True False"}))

if __name__ == "__main__":
    unittest.main()
