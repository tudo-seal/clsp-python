import logging
from typing import TypeVar, Any
import unittest
from clsp import FiniteCombinatoryLogic, enumerate_terms, DSL, Subtypes
from clsp.enumeration import interpret_term
from clsp.types import Constructor

A = Constructor("A")
B = Constructor("B")

class TestList(unittest.TestCase):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(module)s %(levelname)s: %(message)s",
        # level=logging.INFO,
    )

    def setUp(self) -> None:
        exampleDelta: dict[str, list[Any]] = {
            "steps": [2],
        }

        def test_input(step, input) -> bool:
            return True
        
        exampleRepo = {
            "a": A,
            "f": DSL()
                .Use("step", "steps")
                .Use("input", A)
                .With(test_input)
                .In(A ** B),
            "x": B**A,
        }

        self.numTerms = 150
        self.results = enumerate_terms(
            A,
            FiniteCombinatoryLogic(exampleRepo, Subtypes({}), exampleDelta).inhabit(A),
            self.numTerms)

    def test_print(self) -> None:
        #str_algebra = {add: str_add, mul: str_mul, 3: "3", 4: "4", 5: "5"}
        
        counter = 0
        for z in self.results:
            counter += 1
            print(counter)
            #print(z)
        self.assertEqual(counter, self.numTerms)
            
            #string_interpretation = interpret_term(z, str_algebra)
            #evaluation = eval(string_interpretation)
            #cls_eval = interpret_term(z)
            #self.logger.info("%s = %s", string_interpretation, cls_eval)
            #self.assertEqual(evaluation, cls_eval)


if __name__ == "__main__":
    unittest.main()
