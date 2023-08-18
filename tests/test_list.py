import logging
from typing import Any, TypeVar
import unittest
from picls import inhabit_and_interpret
from picls.types import Arrow, Constructor, Type

T = TypeVar("T")


def nil() -> list[Any]:
    return list()


def cons(x: T, xs: list[T]) -> list[T]:
    the_list = [x]
    the_list.extend(xs)
    return the_list


A = Constructor("A")
B = Constructor("B")
C = Constructor("C")


def List(a: Type) -> Constructor:
    return Constructor("List", a)


def a_to_b(a: list[Any]) -> int:
    return len(a)


exampleRepo = {
    nil: List(A),
    cons: Arrow(A, Arrow(List(A), List(A))),
    a_to_b: Arrow(A, B),
    lambda _: "b": Arrow(A, B),
    lambda _: 2: Arrow(A, B),
    lambda f, xs: list(map(f, xs)): Arrow(Arrow(A, B), Arrow(List(A), List(B))),
    "a": A,
}


class TestList(unittest.TestCase):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(module)s %(levelname)s: %(message)s",
        # level=logging.INFO,
    )

    def setUp(self) -> None:
        self.count = 10
        self.results = list(
            inhabit_and_interpret(exampleRepo, List(B), max_count=self.count)
        )
        for z in self.results:
            self.logger.info(z)

    def test_size(self) -> None:
        self.assertEqual(self.count, len(self.results))

    def test_all_monotone(self) -> None:
        max_len = 0
        for z in self.results:
            self.assertGreaterEqual(len(z), max_len)
            max_len = len(z)

    def test_all_lists(self) -> None:
        for z in self.results:
            self.assertIsInstance(z, list)


if __name__ == "__main__":
    unittest.main()
