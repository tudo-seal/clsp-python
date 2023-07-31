from collections.abc import Callable, Sequence
import logging
from typing import Any
import unittest
from cls.combinatorics import minimal_covers, maximal_elements
from itertools import combinations
from collections import deque
from random import randrange


def naive_minimal_covers(
    sets: list[list[int]],
    to_cover: list[int],
    contains: Callable[[list[int], int], bool],
) -> list[list[list[int]]]:
    covers: Any = deque()
    for i in range(len(sets) + 1):
        for cover in combinations(range(len(sets)), i):
            if all(any(contains(sets[s], e) for s in cover) for e in to_cover):
                covers.append(cover)
    covers = list(map(list, covers))
    covers = maximal_elements(covers, lambda c1, c2: all(s in c1 for s in c2))
    return [[sets[i] for i in c] for c in covers]


def equivalent_lists(l1: set[Any], l2: set[Any]) -> bool:
    return len(l1) == len(l2) and all(e in l2 for e in l1) and all(e in l1 for e in l2)


def contains(s: Sequence[Any], e: Any) -> bool:
    return e in s


class Test(unittest.TestCase):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(module)s %(levelname)s: %(message)s",
        # level=logging.INFO,
    )

    def equivalent_covers(self, covers1: Sequence[Any], covers2: Sequence[Any]) -> None:
        return self.assertTrue(
            len(covers1) == len(covers2)
            and all(any(equivalent_lists(c1, c2) for c2 in covers2) for c1 in covers1)
            and all(any(equivalent_lists(c1, c2) for c1 in covers1) for c2 in covers2)
        )

    def test_set_cover1(self):
        # TODO: testing framework
        sets: list[list[int]] = [
            [1, 4],
            [7, 3],
            [7, 9],
            [0, 1, 2],
            [1, 3],
            [2, 3],
            [1, 6],
        ]
        elements: list[int] = [0, 1, 2, 3, 7]
        self.logger.info(minimal_covers(sets, elements, contains))

        # naive minimal set cover implementation

        self.logger.info(naive_minimal_covers(sets, elements, contains))
        self.equivalent_covers(
            minimal_covers(sets, elements, contains),
            naive_minimal_covers(sets, elements, contains),
        )

    def test_set_cover2(self):
        # [7, 8, 9, 5, 3, 4, 1, 4] by
        # [[], [5], [], [5, 5, 0, 0, 1, 4, 5, 4, 1], [5], [7, 7, 9, 3, 7, 0, 7, 7, 0, 3, 8, 4],
        #   [4, 6, 8, 7, 1, 9, 2, 8, 6], [7, 8, 9, 4, 1, 9, 3, 5], [8, 5]]
        elements = [7, 8, 9, 5, 3, 4, 1, 4]
        sets = [
            [],
            [5],
            [],
            [5, 5, 0, 0, 1, 4, 5, 4, 1],
            [5],
            [7, 7, 9, 3, 7, 0, 7, 7, 0, 3, 8, 4],
            [4, 6, 8, 7, 1, 9, 2, 8, 6],
            [7, 8, 9, 4, 1, 9, 3, 5],
            [8, 5],
        ]
        self.logger.info(minimal_covers(sets, elements, contains))
        self.logger.info(naive_minimal_covers(sets, elements, contains))
        self.equivalent_covers(
            minimal_covers(sets, elements, contains),
            naive_minimal_covers(sets, elements, contains),
        )

    def test_set_cover3(self):
        # equivalence exhaustive check
        max_elements = 20
        max_sets = 20

        def random_set() -> list[int]:
            num_elements = randrange(2 * max_elements)
            return [randrange(max_elements) for _ in range(num_elements)]

        sets = [random_set() for _ in range(randrange(max_sets))]
        elements = random_set()
        covers1 = minimal_covers(sets, elements, contains)
        covers2 = naive_minimal_covers(sets, elements, contains)
        self.equivalent_covers(covers1, covers2)


if __name__ == "__main__":
    unittest.main()
