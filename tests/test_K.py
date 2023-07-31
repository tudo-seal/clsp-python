from collections.abc import Callable, Mapping
import logging
import unittest
from cls import (
    Type,
    Constructor,
    Arrow,
    FiniteCombinatoryLogic,
    enumerate_terms,
    Subtypes,
)
from cls.enumeration import interpret_term

X: str = "X"
Y: str = "Y"
K: Callable[[str], Callable[[str], str]] = lambda x: (lambda y: x)


def K2(x: str, y: str) -> str:
    return x


MAP: Callable[[str, Callable[[str], str]], str] = lambda x, f: f(x)


def MAP2(x: str, f: Callable[[str], str]) -> str:
    return f(x)


class TestK(unittest.TestCase):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(module)s %(levelname)s: %(message)s",
        # level=logging.INFO,
    )

    def setUp(self) -> None:
        a: Type = Constructor("a")
        b: Type = Constructor("b")
        c: Type = Constructor("c")
        # d: Type = Constructor("d")

        repository: Mapping[
            str
            | Callable[[str, str], str]
            | Callable[[str], Callable[[str], str]]
            | Callable[[str, Callable[[str], str]], str],
            Type,
        ] = dict(
            {
                X: a,
                Y: b,
                K: Arrow(a, Arrow(b, c)),
                K2: Arrow(a, Arrow(b, c)),
                MAP: Arrow(b, Arrow(Arrow(b, c), c)),
                MAP2: Arrow(b, Arrow(Arrow(b, c), c)),
            }
        )
        environment: dict[str, set[str]] = dict()
        subtypes = Subtypes(environment)

        target = c
        # target = Var(Arrow(b, c))

        fcl = FiniteCombinatoryLogic(repository, subtypes)
        result = fcl.inhabit(target)

        self.enumerated_result = list(enumerate_terms(target, result))

        self.interpreted_terms = []
        for real_result in self.enumerated_result:
            self.logger.info(real_result)
            interpreted_term = interpret_term(real_result)
            self.interpreted_terms.append(interpreted_term)
            self.assertEqual("X", interpreted_term)
            self.logger.info(interpreted_term)

    def test_length(self):
        self.assertEqual(6, len(self.enumerated_result))

    def test_terms(self):
        self.assertIn((K, (("X", ()), ("Y", ()))), self.enumerated_result[:2])
        self.assertIn((K2, (("X", ()), ("Y", ()))), self.enumerated_result[:2])
        self.assertIn((MAP, (("Y", ()), (K, (("X", ()),)))), self.enumerated_result[2:])
        self.assertIn(
            (MAP2, (("Y", ()), (K, (("X", ()),)))), self.enumerated_result[2:]
        )
        self.assertIn(
            (MAP, (("Y", ()), (K2, (("X", ()),)))), self.enumerated_result[2:]
        )
        self.assertIn(
            (MAP2, (("Y", ()), (K2, (("X", ()),)))), self.enumerated_result[2:]
        )

    def test_interpretations(self):
        for x in self.interpreted_terms:
            self.assertEqual("X", x)


if __name__ == "__main__":
    unittest.main()
