import logging
import unittest
from typing import Any
from clsp.dsl import DSL
from clsp import (
    Type,
    Constructor,
    Arrow,
    FiniteCombinatoryLogic,
    enumerate_terms,
    Subtypes,
)

class TestPrune(unittest.TestCase):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(module)s %(levelname)s: %(message)s",
        # level=logging.INFO,
    )

    def setUp(self) -> None:
        a: Type = Constructor("a")
        b: Type = Constructor("b")
        c: Type = Constructor("c")
        d: Type = Constructor("d")

        delta: dict[str, list[Any]] = {
            "empty": [],
        }

        repository = dict(
            {
                # a -> b (cycle)
                "X": Arrow(a, b),
                # b -> a (cycle)
                "Y": Arrow(b, a),
                # c -> b -> a
                "Z": Arrow(c, Arrow(b, a)),
                # d -> c (non-constructible d)
                "W": Arrow(d, c),
                # c
                "C": c,
                # c -> d with unsatisfiable predicate
                "D": DSL()
                .Use("x", c)
                .With(lambda x: False)
                .In(d),
                "E": DSL()
                .Use("i", "empty")
                .In(d),
            }
        )
        environment: dict[str, set[str]] = dict()
        subtypes = Subtypes(environment)

        target = a

        fcl = FiniteCombinatoryLogic(repository, subtypes, delta)
        self.result = fcl.inhabit(target)

        self.enumerated_result = list(enumerate_terms(target, self.result))
        self.expected_nonterminals = frozenset([c, d])

    def test_length(self) -> None:
        self.assertEqual(0, len(self.enumerated_result))

    def test_grammar(self) -> None:
        self.assertEqual(self.expected_nonterminals, frozenset(self.result.nonterminals()))


if __name__ == "__main__":
    unittest.main()
