from collections import defaultdict
import logging
from typing import Any, TypeVar
import unittest
from clsp import FiniteCombinatoryLogic, enumerate_terms
from clsp.dsl import DSL
from clsp.enumeration import interpret_term
from clsp.types import Constructor
from random import random

T = TypeVar("T")


Int = Constructor("Int")


def str_add(a: str, b: str) -> str:
    return f"({a} + {b})"


def str_mul(a: str, b: str) -> str:
    return f"({a} * {b})"


class TestSample(unittest.TestCase):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(module)s %(levelname)s: %(message)s",
        # level=logging.INFO,
    )

    @staticmethod
    def term_size(t: Any) -> int:
        return len(str(t))

    def test_completely_random_sample(self) -> None:
        """
        Randomly reject terms in enumeration.

        Each usage of `str_add` and `str_mul` can be randomly withheld in enumeration
        based on its size.

        The larger the term is, the more likely it is, that it is filtered away.
        Since there are more (possible) larger terms than smaller terms, this should yield some
        sampling.

        This has the disadvantage of starvation. Once no terms of a level are generated, subsequent
        terms will not be generated.

        Note:
          A withheld term does not appear as a sub term in other terms.
        """
        exampleRepo = {
            str_add: DSL()
            .Use("lft", Int)
            .Use("rght", Int)
            .With(lambda lft, rght: random() < 3 / (self.term_size(lft) + self.term_size(rght)))
            .In(Int),
            str_mul: DSL()
            .Use("lft", Int)
            .Use("rght", Int)
            .With(lambda lft, rght: random() < 3 / (self.term_size(lft) + self.term_size(rght)))
            .In(Int),
            "3": Int,
            "4": Int,
            "5": Int,
        }
        results = enumerate_terms(Int, FiniteCombinatoryLogic(exampleRepo).inhabit(Int), 1000)

        for z in results:
            conf_interpretation = interpret_term(z)
            self.logger.info("%s", conf_interpretation)

    def test_non_ideal_but_better_random_sample(self) -> None:
        """
        Randomly reject terms in enumeration, but based on term length.

        Each usage of `str_add` and `str_mul` can be randomly withheld in enumeration
        based on the number of terms already enumerated of that size.

        The more terms of the size were generated, the less likely another term is enumerated.

        This has the disadvantage, that it relies on an outside dictionary for its side-effect.
        While it is not prone to starvation (The probability for the first term of a size is
        always 1), terms earlier in the enumeration have a higher probability to be chosen.
        """
        terms_at_level: dict[int, int] = defaultdict(lambda: 0)

        def random_choose_predicate(lft: Any, rght: Any) -> bool:
            term_size = self.term_size(lft) + self.term_size(rght)
            number_of_terms_at_this_level = terms_at_level[term_size]
            terms_at_level[term_size] += 1
            return random() < 1 / (1 + number_of_terms_at_this_level)

        exampleRepo = {
            str_add: DSL().Use("lft", Int).Use("rght", Int).With(random_choose_predicate).In(Int),
            str_mul: DSL().Use("lft", Int).Use("rght", Int).With(random_choose_predicate).In(Int),
            "3": Int,
            "4": Int,
            "5": Int,
        }
        results = list(enumerate_terms(Int, FiniteCombinatoryLogic(exampleRepo).inhabit(Int), 100))

        for z in results:
            conf_interpretation = interpret_term(z)
            self.logger.info("%s", conf_interpretation)

        self.assertEqual(len(results), 100)


if __name__ == "__main__":
    unittest.main()
