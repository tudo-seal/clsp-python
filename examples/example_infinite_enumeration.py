# Demonstration of the DSL for having non-inferrable infinite parameter spaces

import logging
import unittest
from collections.abc import Iterable, Iterator
from clsp.dsl import DSL
from clsp.synthesizer import Synthesizer
from collections.abc import Container
from clsp.types import Type, Constructor, Var, Literal, Omega
from clsp.inspector import Inspector
from clsp.solution_space import SolutionSpace

class TestDSLUse(unittest.TestCase):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(module)s %(levelname)s: %(message)s",
        # level=logging.INFO,
    )

    def setUp(self) -> None:
        self.inspector = Inspector()

    def test_param(self) -> None:
        # literal varibles can be assigned computed values
        def X(x: int, y: int, t1: str, t2: str) -> str:
            return f"X {x} {y} {t1} {t2}"
        
        def Y(x: int) -> str:
            return f"Y {x}"
        
        def Z(x: int) -> str:
            return f"Z {x}"

        componentSpecifications = {
            X: DSL()
            .Parameter("x", "nat")
            .Parameter("y", "nat")
            .Argument("t1", Var("x"))
            .Argument("t2", Var("y"))
            .Suffix(Constructor("a")),
            Y: DSL()
            .Parameter("x", "nat")
            .Suffix(Var("x")),
            Z: DSL()
            .Parameter("y", "nat")
            .Suffix(Var("y")),
        }

        # infinite enumeration of natural numbers
        class Nat(Iterable[int], Container):
            def __iter__(self) -> Iterator[int]:
                i: int = 0
                while True:
                    yield i
                    i += 1
            def __contains__(self, value: object) -> bool:
                return isinstance(value, int) and value >= 0

        parameterSpace={"nat": Nat()}
        synthesizer = Synthesizer(componentSpecifications, parameterSpace)
        
        target = Constructor("a")
        
        
        solution_space = SolutionSpace()
        for nt, rule in synthesizer.constructSolutionSpaceRules(target):
            solution_space.add_rule(nt, rule.terminal, rule.arguments, rule.predicates)
            for tree in solution_space.enumerate_trees(target):
                 print(tree.interpret())
        
if __name__ == "__main__":
    unittest.main()
