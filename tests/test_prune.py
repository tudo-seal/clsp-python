from collections.abc import Callable
import logging
import unittest
from picls import (
    Type,
    Constructor,
    Arrow,
    FiniteCombinatoryLogic,
    enumerate_terms,
    Subtypes,
)

X: Callable[[str], str] = lambda y: f"X {y}"
Y: Callable[[str], str] = lambda x: f"Y {x}"


class TestPrune(unittest.TestCase):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(module)s %(levelname)s: %(message)s",
        # level=logging.INFO,
    )

    def setUp(self) -> None:
        a: Type = Constructor("a")
        b: Type = Constructor("b")
        # d: Type = Constructor("d")

        repository = dict(
            {
                X: Arrow(a, b),
                Y: Arrow(b, a),
            }
        )
        environment: dict[str, set[str]] = dict()
        subtypes = Subtypes(environment)

        target = a
        # target = Var(Arrow(b, c))

        fcl = FiniteCombinatoryLogic(repository, subtypes)
        self.result = fcl.inhabit(target)

        self.enumerated_result = list(enumerate_terms(target, self.result))

    def test_length(self) -> None:
        self.assertEqual(0, len(self.enumerated_result))

    def test_grammar(self) -> None:
        self.assertEqual(0, len(list(self.result.nonterminals())))


if __name__ == "__main__":
    unittest.main()
