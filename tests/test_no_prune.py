# Demonstration of the DSL for assigning values to lietral variables

#TODO more tests on different variants how values can be created, also infinite contains

import logging
import unittest
from clsp.dsl import DSL
from clsp.synthesizer import Synthesizer
from collections.abc import Container
from clsp.types import Type, Constructor, Var, Literal, Omega, Arrow
from clsp.inspector import Inspector

class TestDSLUse(unittest.TestCase):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(module)s %(levelname)s: %(message)s",
    )

    def setUp(self) -> None:
        self.inspector = Inspector()

    def test_param(self) -> None:
        def AB(s: str) -> str:
            return f"AB {s}"
        def BA(s: str) -> str:
            return f"BA {s}"

        componentSpecifications = {
            # recursive unproductive specification
            AB: Arrow(Constructor("a"), Constructor("b")),
            BA: Arrow(Constructor("b"), Constructor("a"))
        }

        synthesizer = Synthesizer(componentSpecifications)
        
        query = Constructor("a")
        solution_space = synthesizer.constructSolutionSpace(query)
        for tree in solution_space.enumerate_trees(query):
            raise NotImplementedError(f"This should not be reached {tree}")


if __name__ == "__main__":
    unittest.main()
